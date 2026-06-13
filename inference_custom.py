# =====================================================================
# INDO-ARYAN FeVeT HYBRID INFERENCE & DIAGNOSTIC SCRIPT (20-Language Stack)
# =====================================================================

from functools import partial
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List

# Import everything directly from your main training file
import fevet_ia  

# Bind the segmenter inside the fevet_ia namespace to prevent global NameErrors
fevet_ia.segmenter = fevet_ia.LinguisticSegmenter()

# =====================================================================
# Component B: Sanskrit Character LSTM Language Model Class
# =====================================================================
class SanskritCharLM(nn.Module):
    def __init__(self, vocab_size: int, num_features: int, feature_matrix: torch.Tensor,
                 embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 3, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.feature_lookup = nn.Parameter(feature_matrix, requires_grad=False)
        self.feature_proj = nn.Linear(num_features, embed_dim)
        
        # Forward components
        self.lstm_fwd = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc_fwd = nn.Linear(hidden_dim, vocab_size)
        
        # Backward components
        self.lstm_bwd = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc_bwd = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x_fwd, x_bwd):
        # Forward pass
        lens_fwd = (x_fwd != self.pad_idx).sum(dim=1).clamp(min=1).cpu()
        emb_fwd = self.feature_proj(self.feature_lookup[x_fwd])
        packed_fwd = nn.utils.rnn.pack_padded_sequence(emb_fwd, lens_fwd, batch_first=True, enforce_sorted=False)
        out_fwd, _ = self.lstm_fwd(packed_fwd)
        out_fwd, _ = nn.utils.rnn.pad_packed_sequence(out_fwd, batch_first=True, total_length=x_fwd.size(1))
        logits_fwd = self.fc_fwd(out_fwd)
        
        # Backward pass
        lens_bwd = (x_bwd != self.pad_idx).sum(dim=1).clamp(min=1).cpu()
        emb_bwd = self.feature_proj(self.feature_lookup[x_bwd])
        packed_bwd = nn.utils.rnn.pack_padded_sequence(emb_bwd, lens_bwd, batch_first=True, enforce_sorted=False)
        out_bwd, _ = self.lstm_bwd(packed_bwd)
        out_bwd, _ = nn.utils.rnn.pad_packed_sequence(out_bwd, batch_first=True, total_length=x_bwd.size(1))
        logits_bwd = self.fc_bwd(out_bwd)
        
        return logits_fwd, logits_bwd

class LMLexiconDataset(Dataset):
    def __init__(self, lexicon: List[str], vocab, segmenter):
        self.samples = []
        bos_idx, eos_idx = vocab.token_to_idx[vocab.bos_token], vocab.token_to_idx[vocab.eos_token]
        for word in lexicon:
            tokens = segmenter.tokenize(word)
            encoded = vocab.encode(tokens)
            
            x_fwd = [bos_idx] + encoded
            y_fwd = encoded + [eos_idx]
            
            x_bwd = [eos_idx] + encoded[::-1]  # Read from right to left
            y_bwd = encoded[::-1] + [bos_idx]  # Predict leftwards
            
            self.samples.append((
                torch.tensor(x_fwd, dtype=torch.long), torch.tensor(y_fwd, dtype=torch.long),
                torch.tensor(x_bwd, dtype=torch.long), torch.tensor(y_bwd, dtype=torch.long)
            ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def lm_collate_fn(batch, pad_idx):
    x_fwd, y_fwd, x_bwd, y_bwd = zip(*batch)
    xf_p = torch.nn.utils.rnn.pad_sequence(x_fwd, batch_first=True, padding_value=pad_idx)
    yf_p = torch.nn.utils.rnn.pad_sequence(y_fwd, batch_first=True, padding_value=pad_idx)
    xb_p = torch.nn.utils.rnn.pad_sequence(x_bwd, batch_first=True, padding_value=pad_idx)
    yb_p = torch.nn.utils.rnn.pad_sequence(y_bwd, batch_first=True, padding_value=pad_idx)
    return xf_p, yf_p, xb_p, yb_p

@torch.no_grad()
def score_sequence_with_lstm(model: nn.Module, vocab, segmenter, word_list: List[str], device: torch.device) -> float:
    model.eval()
    bos_idx, eos_idx = vocab.token_to_idx[vocab.bos_token], vocab.token_to_idx[vocab.eos_token]
    encoded = vocab.encode(word_list)
    
    x_fwd = torch.tensor([bos_idx] + encoded, dtype=torch.long, device=device).unsqueeze(0)
    y_fwd = torch.tensor(encoded + [eos_idx], dtype=torch.long, device=device).unsqueeze(0)
    
    x_bwd = torch.tensor([eos_idx] + encoded[::-1], dtype=torch.long, device=device).unsqueeze(0)
    y_bwd = torch.tensor(encoded[::-1] + [bos_idx], dtype=torch.long, device=device).unsqueeze(0)
    
    logits_fwd, logits_bwd = model(x_fwd, x_bwd)
    
    log_probs_fwd = torch.log_softmax(logits_fwd, dim=-1)
    gathered_fwd = torch.gather(log_probs_fwd, dim=-1, index=y_fwd.unsqueeze(-1)).squeeze(-1)
    
    log_probs_bwd = torch.log_softmax(logits_bwd, dim=-1)
    gathered_bwd = torch.gather(log_probs_bwd, dim=-1, index=y_bwd.unsqueeze(-1)).squeeze(-1)
    
    # Average the forward and backward log likelihoods
    return (gathered_fwd.sum().item() + gathered_bwd.sum().item()) / 2.0

# =====================================================================
# Diagnostic & Helper Functions
# =====================================================================
def verify_zero_feature_tokens(vocab, feature_matrix):
    zero_feature_tokens = []
    for token, idx in vocab.token_to_idx.items():
        if token in vocab.special_tokens or (token.startswith("[") and token.endswith("]")):
            continue
        feat_sum = feature_matrix[idx, 1:].sum().item()
        if feat_sum == 0.0:
            zero_feature_tokens.append(token)
            
    print("-" * 60)
    print("DIAGNOSTIC: Zero-Feature Token Check")
    print("-" * 60)
    if len(zero_feature_tokens) == 0:
        print("[SUCCESS] All active phonemes have been successfully mapped to phonological features!")
    else:
        print(f"[WARNING] Found {len(zero_feature_tokens)} tokens with no phonological features assigned:")
        print(zero_feature_tokens)
    print("-" * 60 + "\n")

def run_custom_inference(model: nn.Module, lstm_model: nn.Module, vocab, segmenter, target_lang_id: int, device: torch.device, label: str,
                         sanskrit_lexicon: set, lambda_dict: float = 0.75, lambda_lstm: float = 0.12,
                         pali="-", prakrit="-", panjabi="-", sindhi="-", romani="-",
                         kashmiri="-", shina="-", hindi="-", garhwali="-", nepali="-",
                         bhojpuri="-", maithili="-", bengali="-", oriya="-", assamese="-",
                         gujarati="-", marathi="-", konkani="-", sinhala="-", dhivehi="-"):
    
    daughter_inputs = {
        "pali": pali, "prakrit": prakrit, "panjabi": panjabi, "sindhi": sindhi, "romani": romani,
        "kashmiri": kashmiri, "shina": shina, "hindi": hindi, "garhwali": garhwali, "nepali": nepali,
        "bhojpuri": bhojpuri, "maithili": maithili, "bengali": bengali, "oriya": oriya, "assamese": assamese,
        "gujarati": gujarati, "marathi": marathi, "konkani": konkani, "sinhala": sinhala, "dhivehi": dhivehi
    }
    src_tokens, src_langs, src_pos = [], [], []
    
    # Chronological and genetic mapping order matching daughter_columns
    lang_list = [
        "pali", "prakrit", "panjabi", "sindhi", "romani",
        "kashmiri", "shina", "hindi", "garhwali", "nepali",
        "bhojpuri", "maithili", "bengali", "oriya", "assamese",
        "gujarati", "marathi", "konkani", "sinhala", "dhivehi"
    ]
    lang_to_idx = {lang: i + 1 for i, lang in enumerate(lang_list)}
    
    for lang, val in daughter_inputs.items():
        val = str(val).strip()
        if val == "-": continue
        
        normalized_val = fevet_ia.normalize_indo_aryan_orthography(val, is_oia=False)
        lang_id = lang_to_idx[lang]
        tokens = segmenter.tokenize(normalized_val)
        encoded_tokens = vocab.encode(tokens)

        # --- MATCH SCRIPT 1 DATASET PREPARATION EXACTLY ---
        # 1. Append SEP to the END of every cognate sequence
        encoded_tokens.append(vocab.token_to_idx[vocab.sep_token])
        
        src_tokens.extend(encoded_tokens)
        src_langs.extend([lang_id] * len(encoded_tokens))
        
        # 2. Assign positions 0 to N. The SEP token cleanly gets position N.
        src_pos.extend(list(range(len(encoded_tokens)))) 
        
    if not src_tokens:
        print(f"No active reflexes found for custom inference: {label}")
        return
        
    # Convert directly to tensors without wrapping in BOS/EOS or shifting positions
    src_tokens_tensor = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)
    src_langs_tensor = torch.tensor(src_langs, dtype=torch.long, device=device).unsqueeze(0)
    src_pos_tensor = torch.tensor(src_pos, dtype=torch.long, device=device).unsqueeze(0)
    
    predictions = fevet_ia.beam_search_decode(
        model, src_tokens_tensor, src_langs_tensor, src_pos_tensor, vocab, target_lang_id, beam_width=20
    )
    
    reranked_predictions = []
    for pred_word, raw_seq_score in predictions:
        joined_word = "".join(pred_word)
        seq_len = len(pred_word) + 1
        
        norm_seq_score = raw_seq_score / (seq_len ** 0.75)
        dict_score = 0.0 if joined_word in sanskrit_lexicon else -lambda_dict
        raw_lstm_score = score_sequence_with_lstm(lstm_model, vocab, segmenter, pred_word, device)
        norm_lstm_score = raw_lstm_score / seq_len
        joint_score = norm_seq_score + dict_score + (lambda_lstm * norm_lstm_score)
        
        reranked_predictions.append((pred_word, norm_seq_score, dict_score, norm_lstm_score, joint_score))
        
    reranked_predictions.sort(key=lambda x: x[4], reverse=True)
    
    print("=" * 60)
    print(f"RECONSTRUCTION TEST: {label}")
    print("=" * 60)
    print("Input Reflexes:")
    for lang, val in daughter_inputs.items():
        if val != "-": print(f"  {lang.upper():10}: {val}")
        
    print("\nTop Sanskrit Hypotheses (With Joint Lexical & Phonotactic Reranker):")
    for b_idx, (pred_word, seq_score, dict_score, lstm_score, joint_score) in enumerate(reranked_predictions[:3]):
        in_dict_str = "[IN DICTIONARY]" if "".join(pred_word) in sanskrit_lexicon else "[NOT IN DICTIONARY]"
        print(f"  Beam {b_idx + 1} (Joint: {joint_score:.3f} | Seq: {seq_score:.3f} | Dict: {dict_score:.1f} | LM: {lstm_score:.2f}) {in_dict_str}: {' '.join(pred_word)}")
    print("=" * 60 + "\n")


# =====================================================================
# Main Execution Block
# =====================================================================

if __name__ == "__main__":
    from sklearn.model_selection import GroupShuffleSplit
    import pandas as pd
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing Inference Script on: {device}")

    target_column = "sanskrit"
    daughter_columns = [
        "pali", "prakrit",                        # Middle Indo-Aryan 
        "panjabi", "sindhi", "romani",            # Northwestern
        "kashmiri", "shina",                      # Dardic
        "hindi", "garhwali", "nepali",            # Central & Pahari
        "bhojpuri", "maithili",                   # Bihari / Eastern Bridge
        "bengali", "oriya", "assamese",           # Eastern
        "gujarati", "marathi", "konkani",         # Western / Southern
        "sinhala", "dhivehi"                      # Insular
    ]
    checkpoint_path = "best_fevet_ia_model.pt"
    lstm_checkpoint_path = "sanskrit_lstm_prior.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find model checkpoint '{checkpoint_path}'. Train the model first.")

    # 1. Download Jambu & preprocess
    print("Retrieving dataset...")
    forms_raw, langs_raw, params_raw = fevet_ia.download_jambu_data()
    df = fevet_ia.process_jambu_to_dataframe_unified(forms_raw, langs_raw, params_raw, daughter_columns)
    df = df.reset_index(drop=True)

    # 2. SPLIT FIRST
    print("Creating dataset and validation splits using Parameter_ID hashing...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=67)
    train_idx, val_idx = next(gss.split(df, groups=df['Parameter_ID']))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # 3. BUILD VOCAB
    print("\nBuilding Reconstruction Vocabulary...")
    train_raw_tokens = []
    for col in [target_column] + daughter_columns:
        for value in train_df[col].dropna():
            if isinstance(value, list):
                for v in value:
                    if str(v) != "-":
                        train_raw_tokens.append(fevet_ia.segmenter.tokenize(str(v)))
            else:
                if str(value) != "-":
                    train_raw_tokens.append(fevet_ia.segmenter.tokenize(str(value)))

    vocab = fevet_ia.ReconstructionVocab()
    vocab.build_vocab(train_raw_tokens, [])
    print(f"Total Unique Vocabulary Tokens: {len(vocab)}")

    pad_idx = vocab.token_to_idx["<PAD>"]

    # 4. Instantiate Datasets
    train_dataset = fevet_ia.JambuCognateDataset(train_df, target_column, daughter_columns, vocab, fevet_ia.segmenter)
    val_dataset = fevet_ia.JambuCognateDataset(val_df, target_column, daughter_columns, vocab, fevet_ia.segmenter)

    # Reconstruct the Sanskrit lexicons
    print("Compiling Sanskrit Lexicon Sets...")
    
    sanskrit_lexicon_full = set()
    for word in df['sanskrit'].dropna():
        tokens = fevet_ia.segmenter.tokenize(str(word))
        cleaned_word = "".join(tokens)
        if cleaned_word and cleaned_word != "-":
            sanskrit_lexicon_full.add(cleaned_word)
            
    headword_col = None
    for col in ['Name', 'Form', 'Value', 'ID']:
        if col in params_raw.columns:
            headword_col = col
            break
    if headword_col:
        for word in params_raw[headword_col].dropna():
            normalized_word = fevet_ia.normalize_indo_aryan_orthography(str(word), is_oia=True)
            tokens = fevet_ia.segmenter.tokenize(normalized_word)
            cleaned_word = "".join(tokens)
            if cleaned_word and cleaned_word != "-":
                sanskrit_lexicon_full.add(cleaned_word)
                
    sanskrit_lexicon_train = set()
    for _, row in train_df.iterrows():
        target_val = str(row['sanskrit']).strip()
        if target_val != "-" and target_val != "":
            tokens = fevet_ia.segmenter.tokenize(target_val)
            clean_w = "".join([t for t in tokens if t not in vocab.special_tokens])
            if clean_w:
                sanskrit_lexicon_train.add(clean_w)

    print(f"Full Dictionary Lexicon: {len(sanskrit_lexicon_full)} words.")
    print(f"LSTM Training Lexicon (No Leakage): {len(sanskrit_lexicon_train)} words.")

    # 3. Build phonetic feature matrix
    feature_matrix = fevet_ia.build_phonological_matrix(vocab).to(device)
    verify_zero_feature_tokens(vocab, feature_matrix.cpu())

    # 4. Instantiate model architecture
    model = fevet_ia.FeVeTTransformer(
        vocab_size=len(vocab),
        num_features=34,
        feature_matrix=feature_matrix,
        num_languages=len(daughter_columns),
        d_model=128,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=2,
        dim_feedforward=256
    ).to(device)

    # 5. Load saved checkpoint weights
    print(f"Loading weights from checkpoint: {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    clean_state_dict = {
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v 
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state_dict)
    model.eval()
    print("Model loaded successfully in evaluation mode.\n")

    # 6. Initialize and Load/Train Component B (Sanskrit Char LSTM)
    lstm_model = SanskritCharLM(
        vocab_size=len(vocab),
        num_features=feature_matrix.shape[1],
        feature_matrix=feature_matrix,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        pad_idx=pad_idx
    ).to(device)
    
    if os.path.exists(lstm_checkpoint_path):
        print(f"Loading pre-trained Sanskrit Character LSTM from '{lstm_checkpoint_path}'...")
        lstm_state_dict = torch.load(lstm_checkpoint_path, map_location=device)
        clean_lstm_state_dict = {
            k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v 
            for k, v in lstm_state_dict.items()
        }
        lstm_model.load_state_dict(clean_lstm_state_dict)
    else:
        print("\n[INFO] No pre-trained Sanskrit LSTM found. Commencing training...")
        lm_dataset = LMLexiconDataset(list(sanskrit_lexicon_train), vocab, fevet_ia.segmenter)
        lm_loader = DataLoader(lm_dataset, batch_size=64, shuffle=True, collate_fn=lambda b: lm_collate_fn(b, pad_idx))
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.98))
        lm_epochs = 30
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, steps_per_epoch=len(lm_loader), epochs=lm_epochs, pct_start=0.2, anneal_strategy='cos')
        
        lstm_model.train()
        for lm_epoch in range(1, lm_epochs + 1):
            total_lm_loss = 0.0
            for xf, yf, xb, yb in lm_loader:
                xf, yf, xb, yb = xf.to(device), yf.to(device), xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits_fwd, logits_bwd = lstm_model(xf, xb)
                loss_fwd = criterion(logits_fwd.reshape(-1, logits_fwd.size(-1)), yf.reshape(-1))
                loss_bwd = criterion(logits_bwd.reshape(-1, logits_bwd.size(-1)), yb.reshape(-1))
                loss = loss_fwd + loss_bwd
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_lm_loss += loss.item()
            print(f"  LSTM Training Epoch {lm_epoch}/{lm_epochs} | Loss: {total_lm_loss / len(lm_loader):.4f}")
            
        torch.save(lstm_model.state_dict(), lstm_checkpoint_path)
        
    lstm_model.eval()

    target_lang_id = len(daughter_columns) + 1

    # =====================================================================
    # UNSEEN VALIDATION SET DIAGNOSTIC CASES (GroupShuffleSplit Seed: 67)
    # =====================================================================

    # 1. Tooth (CDIAL 6152: danta)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Tooth (Expected Skt: danta)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="danta",
        prakrit="daṁta",
        panjabi="dand",
        sindhi="ᶑandu",
        romani="dand",
        kashmiri="dand",
        shina="dɔn",
        hindi="dā̃t",
        garhwali="dā̃t",
        nepali="-",
        bhojpuri="dā̃t",
        maithili="dā̃t",
        bengali="dā̃t",
        oriya="dānta",
        assamese="dā̃t",
        gujarati="dā̃t",
        marathi="dā̃t",
        konkani="dāntu",
        sinhala="data",
        dhivehi="dat"
    )

    # 2. Thirteen (CDIAL 6001: trayōdaśa)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Thirteen (Expected Skt: trayōdaśa)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="tēlasa",
        prakrit="tidaśa",
        panjabi="tērʱã",
        sindhi="terãhã",
        romani="-",
        kashmiri="truvāh",
        shina="ʦ̣õi",
        hindi="tērā",
        garhwali="tera",
        nepali="-",
        bhojpuri="terah",
        maithili="terah",
        bengali="tera",
        oriya="tera",
        assamese="tera",
        gujarati="tera",
        marathi="terā",
        konkani="terā",
        sinhala="teḷesa",
        dhivehi="tera"
    )

    # 3. Thirteen Variant (CDIAL 6001: *trayēdaśa)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Thirteen Variant (Expected Skt: *trayēdaśa)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="tēlasa",
        prakrit="tidaśa",
        panjabi="tērʱã",
        sindhi="terãhã",
        romani="-",
        kashmiri="truvāh",
        shina="ʦ̣õi",
        hindi="tērā",
        garhwali="tera",
        nepali="-",
        bhojpuri="terah",
        maithili="terah",
        bengali="tera",
        oriya="tera",
        assamese="tera",
        gujarati="tera",
        marathi="terā",
        konkani="terā",
        sinhala="teḷesa",
        dhivehi="tera"
    )

    # 4. Fourteen Variant (CDIAL 4605: *catrudaśa)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Fourteen Variant (Expected Skt: *catrudaśa)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="catuddasa",
        prakrit="codasa",
        panjabi="cɔdā̃",
        sindhi="coᶑãhã",
        romani="-",
        kashmiri="ʦɔdāh",
        shina="condai",
        hindi="caudah",
        garhwali="cɔdda",
        nepali="-",
        bhojpuri="caudah",
        maithili="caudah",
        bengali="codda",
        oriya="cauda",
        assamese="saidʱy",
        gujarati="cauda",
        marathi="ʦaudā",
        konkani="coudā",
        sinhala="sudusa",
        dhivehi="sauda"
    )

    # 5. Thou (CDIAL 5889: tuvam)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Thou (Expected Skt: tuvam)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="tuvaṁ",
        prakrit="tu",
        panjabi="tũ",
        sindhi="tū̃",
        romani="tu",
        kashmiri="tu",
        shina="tu",
        hindi="tūṁ",
        garhwali="tu",
        nepali="-",
        bhojpuri="tu",
        maithili="-",
        bengali="tu",
        oriya="tu",
        assamese="tuhã",
        gujarati="tū̃",
        marathi="tū̃",
        konkani="tū",
        sinhala="tō",
        dhivehi="ta"
    )

    # 6. One (CDIAL 2462: ēka)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="One (Expected Skt: ēka)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="ēka",
        prakrit="eka",
        panjabi="hekk",
        sindhi="eku",
        romani="ak",
        kashmiri="yēku",
        shina="yēkaṭik",
        hindi="ēk",
        garhwali="ēk",
        nepali="-",
        bhojpuri="ego",
        maithili="ek",
        bengali="ek",
        oriya="e",
        assamese="eṭā",
        gujarati="ek",
        marathi="ek",
        konkani="-",
        sinhala="eka",
        dhivehi="ek"
    )

    # 7. Twelve Variant (CDIAL 6658: duvādaśa)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Twelve Variant (Expected Skt: duvādaśa)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="dvādasa",
        prakrit="badaya",
        panjabi="bārʱã",
        sindhi="ɓārãhã",
        romani="-",
        kashmiri="bāh",
        shina="bāi",
        hindi="bārā",
        garhwali="bāra",
        nepali="-",
        bhojpuri="bārē",
        maithili="bārah",
        bengali="bāra",
        oriya="bāra",
        assamese="bāra",
        gujarati="bār",
        marathi="bārā",
        konkani="bārā",
        sinhala="doḷasa",
        dhivehi="bāra"
    )

    # 8. Fourteen (CDIAL 4605: caturdaśa)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Fourteen (Expected Skt: caturdaśa)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="catuddasa",
        prakrit="codasa",
        panjabi="cɔdā̃",
        sindhi="coᶑãhã",
        romani="-",
        kashmiri="ʦɔdāh",
        shina="condai",
        hindi="caudah",
        garhwali="cɔdda",
        nepali="-",
        bhojpuri="caudah",
        maithili="caudah",
        bengali="codda",
        oriya="cauda",
        assamese="saidʱy",
        gujarati="cauda",
        marathi="ʦaudā",
        konkani="coudā",
        sinhala="sudusa",
        dhivehi="sauda"
    )

    # 9. One Variant (CDIAL 2462: *ēkka)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="One Variant (Expected Skt: *ēkka)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="ēka",
        prakrit="eka",
        panjabi="hekk",
        sindhi="eku",
        romani="ak",
        kashmiri="yēku",
        shina="yēkaṭik",
        hindi="ēk",
        garhwali="ēk",
        nepali="-",
        bhojpuri="ego",
        maithili="ek",
        bengali="ek",
        oriya="e",
        assamese="eṭā",
        gujarati="ek",
        marathi="ek",
        konkani="-",
        sinhala="eka",
        dhivehi="ek"
    )

    # 10. Twelve (CDIAL 6658: dvādaśa)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Twelve (Expected Skt: dvādaśa)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="dvādasa",
        prakrit="badaya",
        panjabi="bārʱã",
        sindhi="ɓārãhã",
        romani="-",
        kashmiri="bāh",
        shina="bāi",
        hindi="bārā",
        garhwali="bāra",
        nepali="-",
        bhojpuri="bārē",
        maithili="bārah",
        bengali="bāra",
        oriya="bāra",
        assamese="bāra",
        gujarati="bār",
        marathi="bārā",
        konkani="bārā",
        sinhala="doḷasa",
        dhivehi="bāra"
    )

    # 11. Knot / Protuberance (CDIAL 4354: grantʰi)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Knot / Protuberance (Expected Skt: grantʰi)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="gaṇṭʰi",
        prakrit="gaṁṭʰi",
        panjabi="ɠaṇḍʱ",
        sindhi="ɠaṇḍʱi",
        romani="-",
        kashmiri="ganḍ",
        shina="guṇ",
        hindi="gāṁṭʰi",
        garhwali="-",
        nepali="-",
        bhojpuri="gā̃ṭʰ",
        maithili="gā̃ṭʰi",
        bengali="gā̃ṭʰ",
        oriya="gaṇṭʰi",
        assamese="gā̃ṭʰi",
        gujarati="gā̃ṭʰ",
        marathi="gā̃ṭʰ",
        konkani="gāṇṭi",
        sinhala="gæṭaya",
        dhivehi="gor̆"
    )

    # 12. Arise / Originate (CDIAL 1814: utpādayati)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Arise / Originate (Expected Skt: utpādayati)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="uppajjati",
        prakrit="upajeśadi",
        panjabi="upajṇā",
        sindhi="upaʄaṇu",
        romani="-",
        kashmiri="vɔpazun",
        shina="-",
        hindi="upajai",
        garhwali="upjaṇu",
        nepali="-",
        bhojpuri="upajal",
        maithili="upajab",
        bengali="upajā",
        oriya="upajibā",
        assamese="opaziba",
        gujarati="ūpajai",
        marathi="upaʣṇẽ",
        konkani="ubjatā",
        sinhala="upadinavā",
        dhivehi="ufedenī"
    )

    # 13. Stand up / Set out (CDIAL 8607: pratiṣṭʰati)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Stand up / Set out (Expected Skt: pratiṣṭʰati)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="paṭṭʰāya",
        prakrit="pattʰaṁta",
        panjabi="paṭṭʰaṇ",
        sindhi="paṭʰaṇu",
        romani="prast",
        kashmiri="paṭʰun",
        shina="-",
        hindi="paṭʰavab",
        garhwali="pātʰṇu",
        nepali="-",
        bhojpuri="paṭʰāval",
        maithili="paṭʰāvai",
        bengali="paṭʰāna",
        oriya="paṭʰāibā",
        assamese="paṭʰāiba",
        gujarati="paraṭʰvũ",
        marathi="pāṭʰaviṇẽ",
        konkani="-",
        sinhala="paṭā",
        dhivehi="far̆an"
    )

    # 14. Put aside / Send (CDIAL 8607: prastʰāpayati)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Put aside / Send (Expected Skt: prastʰāpayati)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="paṭṭʰāya",
        prakrit="pattʰaṁta",
        panjabi="paṭṭʰaṇ",
        sindhi="paṭʰaṇu",
        romani="prast",
        kashmiri="paṭʰun",
        shina="-",
        hindi="paṭʰavab",
        garhwali="pātʰṇu",
        nepali="-",
        bhojpuri="paṭʰāval",
        maithili="paṭʰāvai",
        bengali="paṭʰāna",
        oriya="paṭʰāibā",
        assamese="paṭʰāiba",
        gujarati="paraṭʰvũ",
        marathi="pāṭʰaviṇẽ",
        konkani="-",
        sinhala="paṭā",
        dhivehi="far̆an"
    )

    # 15. Produced / Born (CDIAL 1814: utpanna)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Produced / Born (Expected Skt: utpanna)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="uppajjati",
        prakrit="upajeśadi",
        panjabi="upajṇā",
        sindhi="upaʄaṇu",
        romani="-",
        kashmiri="vɔpazun",
        shina="-",
        hindi="upajai",
        garhwali="upjaṇu",
        nepali="-",
        bhojpuri="upajal",
        maithili="upajab",
        bengali="upajā",
        oriya="upajibā",
        assamese="opaziba",
        gujarati="ūpajai",
        marathi="upaʣṇẽ",
        konkani="ubjatā",
        sinhala="upadinavā",
        dhivehi="ufedenī"
    )

    # 16. Arise / Originate Variant (CDIAL 1814: utpadyatē)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Arise / Originate Variant (Expected Skt: utpadyatē)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="uppajjati",
        prakrit="upajeśadi",
        panjabi="upajṇā",
        sindhi="upaʄaṇu",
        romani="-",
        kashmiri="vɔpazun",
        shina="-",
        hindi="upajai",
        garhwali="upjaṇu",
        nepali="-",
        bhojpuri="upajal",
        maithili="upajab",
        bengali="upajā",
        oriya="upajibā",
        assamese="opaziba",
        gujarati="ūpajai",
        marathi="upaʣṇẽ",
        konkani="ubjatā",
        sinhala="upadinavā",
        dhivehi="ufedenī"
    )

    # 17. Play / Sport (CDIAL 3918: *kʰēḍḍ)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Play / Sport (Expected Skt: *kʰēḍḍ)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="kʰiḍḍā",
        prakrit="kʰēḍaṇaa",
        panjabi="kʰeḍaṇ",
        sindhi="kʰeᶑaṇu",
        romani="kʰel",
        kashmiri="kʰēlun",
        shina="-",
        hindi="kʰillī",
        garhwali="kʰeḷnu",
        nepali="-",
        bhojpuri="kʰēlal",
        maithili="kʰelab",
        bengali="kʰelā",
        oriya="kʰeɽa",
        assamese="kʰelāiba",
        gujarati="kʰeḍɔ",
        marathi="kʰeḷṇẽ",
        konkani="kʰeḷtā",
        sinhala="keḷanavā",
        dhivehi="-"
    )

    # 18. Fly / Fall (CDIAL 7722: *patta-)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Fly / Fall (Expected Skt: *patta-)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="patati",
        prakrit="paḍai",
        panjabi="pɛvaṇ",
        sindhi="pavaṇu",
        romani="per",
        kashmiri="pyonu",
        shina="poiki̯",
        hindi="parab",
        garhwali="paɽnu",
        nepali="-",
        bhojpuri="paral",
        maithili="parab",
        bengali="paɽā",
        oriya="paɽibā",
        assamese="pariba",
        gujarati="paɽvũ",
        marathi="paḍṇẽ",
        konkani="paḍtā",
        sinhala="-",
        dhivehi="-"
    )

    # 19. Play / Sport Variant (CDIAL 3918: *kʰiḍḍ-)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Play / Sport Variant (Expected Skt: *kʰiḍḍ-)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="kʰiḍḍā",
        prakrit="kʰēḍaṇaa",
        panjabi="kʰeḍaṇ",
        sindhi="kʰeᶑaṇu",
        romani="kʰel",
        kashmiri="kʰēlun",
        shina="-",
        hindi="kʰillī",
        garhwali="kʰeḷnu",
        nepali="-",
        bhojpuri="kʰēlal",
        maithili="kʰelab",
        bengali="kʰelā",
        oriya="kʰeɽa",
        assamese="kʰelāiba",
        gujarati="kʰeḍɔ",
        marathi="kʰeḷṇẽ",
        konkani="kʰeḷtā",
        sinhala="keḷanavā",
        dhivehi="-"
    )

    # 20. Corrosive / Alkali (CDIAL 3674: kṣāra)
    run_custom_inference(
        model, lstm_model, vocab, fevet_ia.segmenter, target_lang_id, device,
        label="Corrosive / Alkali (Expected Skt: kṣāra)", sanskrit_lexicon=sanskrit_lexicon_full, lambda_lstm=0.15,
        pali="kʰāra",
        prakrit="kʰāra",
        panjabi="kʰār",
        sindhi="kʰāru",
        romani="car",
        kashmiri="kʰāra",
        shina="-",
        hindi="kʰārā",
        garhwali="kʰāru",
        nepali="-",
        bhojpuri="kʰār",
        maithili="kʰariā",
        bengali="kʰālāɽi",
        oriya="kʰāra",
        assamese="kʰār",
        gujarati="kʰāra",
        marathi="kʰār",
        konkani="kʰāru",
        sinhala="kara",
        dhivehi="-"
    )