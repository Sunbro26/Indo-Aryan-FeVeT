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

def calculate_levenshtein(seq1: List[str], seq2: List[str]) -> int:
    """Computes the Levenshtein distance between two token lists."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]
    
    for x in range(size_x): matrix[x][0] = x
    for y in range(size_y): matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = min(matrix[x-1][y] + 1, matrix[x][y-1] + 1, matrix[x-1][y-1])
            else:
                matrix[x][y] = min(matrix[x-1][y] + 1, matrix[x][y-1] + 1, matrix[x-1][y-1] + 1)
    return matrix[size_x - 1][size_y - 1]

def run_hybrid_validation(model, lstm_model, val_df, vocab, segmenter, daughter_columns, target_lang_id, device,
                          sanskrit_lexicon, lambda_dict=0.75, lambda_lstm=0.12, beam_width=10):
    """
    Evaluates the whole validation set by injecting test samples into Beam Search 
    and re-ranking them using the combined FeVeT logprobs, LSTM prior, and CDIAL dictionary prior.
    """
    print("\n" + "=" * 60)
    print(f"RUNNING HYBRID EVALUATION ON VALIDATION SET (Beam={beam_width})")
    print("This will take a while. Sit tight...")
    print("=" * 60)
    
    start_time = time.time()
    exact_matches = 0
    fuzzy_1 = 0
    fuzzy_2 = 0
    total_ld = 0
    valid_samples = 0
    
    # Genetic/Chronological Language Map
    lang_to_idx = {lang.lower(): i + 1 for i, lang in enumerate(daughter_columns)}
    
    for idx, row in val_df.iterrows():
        gold_target = str(row['sanskrit']).strip()
        if gold_target == "-" or gold_target.lower() == "nan" or not gold_target:
            continue
            
        gold_tokens = segmenter.tokenize(gold_target)
        gold_clean = [t for t in gold_tokens if t not in vocab.special_tokens]
        
        src_tokens, src_langs, src_pos = [], [], []
        
        for lang in daughter_columns:
            val = row[lang]
            val_list = val if isinstance(val, list) else [val]
            
            for v in val_list:
                v_str = str(v).strip()
                if v_str == "-" or v_str.lower() == "nan" or not v_str: 
                    continue
                
                normalized_v = fevet_ia.normalize_indo_aryan_orthography(v_str, is_oia=False)
                tokens = segmenter.tokenize(normalized_v)
                encoded_tokens = vocab.encode(tokens)
                
                # --- MATCH SCRIPT 1 DATASET PREPARATION EXACTLY ---
                # 1. Append SEP to the END of every cognate sequence
                encoded_tokens.append(vocab.token_to_idx[vocab.sep_token])
                
                src_tokens.extend(encoded_tokens)
                src_langs.extend([lang_to_idx[lang.lower()]] * len(encoded_tokens))
                
                # 2. Assign positions 0 to N. The SEP token cleanly gets position N.
                src_pos.extend(list(range(len(encoded_tokens))))
                
        if not src_tokens:
            continue
            
        # Do not wrap the source in BOS/EOS or shift the position values
        src_tokens_tensor = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)
        src_langs_tensor = torch.tensor(src_langs, dtype=torch.long, device=device).unsqueeze(0)
        src_pos_tensor = torch.tensor(src_pos, dtype=torch.long, device=device).unsqueeze(0)
        
        # Execute Augmented Beam Search
        predictions = fevet_ia.beam_search_decode(
            model, src_tokens_tensor, src_langs_tensor, src_pos_tensor, 
            vocab, target_lang_id, beam_width=beam_width
        )
        
        if not predictions: continue
            
        # Rerank Predictions using Dictionary & Bidirectional LSTM Prior
        reranked_predictions = []
        for pred_word, raw_seq_score in predictions:
            joined_word = "".join(pred_word)
            seq_len = len(pred_word) + 1
            
            norm_seq_score = raw_seq_score / (seq_len ** 0.75)
            dict_score = 0.0 if joined_word in sanskrit_lexicon else -lambda_dict
            raw_lstm_score = score_sequence_with_lstm(lstm_model, vocab, segmenter, pred_word, device)
            norm_lstm_score = raw_lstm_score / seq_len
            joint_score = norm_seq_score + dict_score + (lambda_lstm * norm_lstm_score)
            
            reranked_predictions.append((pred_word, joint_score))
            
        reranked_predictions.sort(key=lambda x: x[1], reverse=True)
        best_pred_tokens = reranked_predictions[0][0]
        
        # Levenshtein Evaluation
        ld = calculate_levenshtein(gold_clean, best_pred_tokens)
        
        if ld == 0: exact_matches += 1
        if ld <= 1: fuzzy_1 += 1
        if ld <= 2: fuzzy_2 += 1
        total_ld += ld
        valid_samples += 1
        
        if valid_samples % 50 == 0:
            print(f"  Processed: {valid_samples}/{len(val_df)} valid samples | "
                  f"Exact: {(exact_matches/valid_samples)*100:.2f}% | "
                  f"LD: {total_ld/valid_samples:.4f}")

    if valid_samples == 0:
        print("No valid samples evaluated.")
        return

    acc = exact_matches / valid_samples
    f1 = fuzzy_1 / valid_samples
    f2 = fuzzy_2 / valid_samples
    avg_ld = total_ld / valid_samples
    mins_taken = (time.time() - start_time) / 60.0
    
    print("\n" + "=" * 60)
    print(f"HYBRID BEAM + LSTM VALIDATION METRICS (FINAL)")
    print(f"Evaluated in {mins_taken:.2f} minutes.")
    print("=" * 60)
    print(f"  Total Valid Samples    : {valid_samples}")
    print(f"  Exact Match Accuracy   : {acc * 100:.2f}%")
    print(f"  Fuzzy-1 Accuracy (LD<=1): {f1 * 100:.2f}%")
    print(f"  Fuzzy-2 Accuracy (LD<=2): {f2 * 100:.2f}%")
    print(f"  Avg Levenshtein Dist   : {avg_ld:.4f}")
    print("=" * 60 + "\n")

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

    val_collate = partial(fevet_ia.collate_fn, pad_idx=pad_idx, lang_dropout_prob=0.0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, collate_fn=val_collate, pin_memory=True, num_workers=4, persistent_workers=True)
    
    # 7. Evaluate Using Raw / Standard Evaluation (Greedy Baseline typically)
    print("Running Baseline validation evaluation on loaded best checkpoint...")
    val_acc, val_ld, val_fld, val_f1, val_f2, pred_sample, gold_sample, stem_3 = fevet_ia.evaluate_reconstructions(
        model, val_loader, vocab, feature_matrix, len(daughter_columns) + 1, device
    )
    
    print("\n" + "=" * 60)
    print("STANDARD EVALUATION METRICS (Baseline Inference)")
    print("=" * 60)
    print(f"  Exact Match Accuracy   : {val_acc * 100:.2f}%")
    print(f"  Fuzzy-1 Accuracy (LD<=1): {val_f1 * 100:.2f}%")
    print(f"  Fuzzy-2 Accuracy (LD<=2): {val_f2 * 100:.2f}%")
    print(f"  Levenshtein Distance   : {val_ld:.4f}")
    print(f"  Feature Distance (FLD) : {val_fld:.4f}")
    print(f"  Stem-3 Match Rate      : {stem_3 * 100:.2f}%")
    print("=" * 60 + "\n")

    target_lang_id = len(daughter_columns) + 1

    # 8. EXECUTE HYBRID INFERENCE OVER ENTIRE VALIDATION SET
    # Beam width of 10 gives an excellent accuracy vs. speed trade-off for full datasets.
    run_hybrid_validation(
        model=model, 
        lstm_model=lstm_model, 
        val_df=val_df, 
        vocab=vocab, 
        segmenter=fevet_ia.segmenter, 
        daughter_columns=daughter_columns,
        target_lang_id=target_lang_id, 
        device=device, 
        sanskrit_lexicon=sanskrit_lexicon_full, 
        lambda_dict=0.75, 
        lambda_lstm=0.15,
        beam_width=20 
    )