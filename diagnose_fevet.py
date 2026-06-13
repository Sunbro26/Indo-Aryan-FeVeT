import torch
import pandas as pd
import os
from typing import List
from sklearn.model_selection import GroupShuffleSplit

# Import all building blocks from your original file
from fevet_ia import (
    FeVeTTransformer, 
    ReconstructionVocab, 
    LinguisticSegmenter, 
    build_phonological_matrix, 
    download_jambu_data, 
    process_jambu_to_dataframe_unified,
    get_causal_mask
)

class FeVeTDiagnosticSuite:
    def __init__(self, model, vocab, segmenter, feature_matrix):
        self.model = model
        self.vocab = vocab
        self.segmenter = segmenter
        self.feature_matrix = feature_matrix
        # Handle torch.compile wrapper
        self.underlying_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    def print_mask_grid(self, mask_tensor: torch.Tensor, labels: List[str], title: str):
        """Prints a 2D boolean mask as a readable binary grid."""
        print(f"\n--- {title} ---")
        
        # Handle dimensions (Batch, Heads, Seq, Seq) -> (Seq, Seq)
        if mask_tensor.dim() == 4:
            mask_tensor = mask_tensor[0, 0]
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor[0]
            
        mask_np = mask_tensor.cpu().numpy().astype(int)
        
        # Create header with truncated labels
        header = " " * 12 + " ".join([f"{l[:3]}" for l in labels])
        print(header)
        print("-" * len(header))
        
        for i, row in enumerate(mask_np):
            row_str = " ".join(["1" if val == 1 else "0" for val in row])
            print(f"{labels[i][:10]:<12} {row_str}")
        print("\n(0 = Allowed Attention, 1 = Masked/Blocked)")

    def run_trace(self, df: pd.DataFrame, daughter_cols: List[str], num_rows: int = 10):
        print("\n" + "="*80)
        print("FEVET-IA TENSOR FLOW DIAGNOSTICS (WITH SPECIAL TOKENS)")
        print("="*80)

        sample_df = df.head(num_rows)
        print("\n[STEP 1] RAW DATA (Wide Format)")
        print(sample_df[['sanskrit'] + daughter_cols])

        print("\n[STEP 2] TOKENIZATION & ID MAPPING")
        row = sample_df.iloc[0]
        
        # Get special token IDs
        bos_id = self.vocab.token_to_idx[self.vocab.bos_token]
        eos_id = self.vocab.token_to_idx[self.vocab.eos_token]
        sep_id = self.vocab.token_to_idx[self.vocab.sep_token]
        pad_id = self.vocab.token_to_idx[self.vocab.pad_token]
        unk_id = self.vocab.token_to_idx[self.vocab.unk_token]

        src_tokens_raw, src_langs_raw, labels = [], [], []
        
        for lang in daughter_cols:
            vals = row[lang]
            if not isinstance(vals, list): vals = [vals]
            for v in vals:
                if v == "-" or v == "": continue
                
                # Let's inject a fake unknown character to test <UNK>
                if lang == "prakrit" and len(src_tokens_raw) > 0 and "<UNK>" not in labels:
                    v = v + "𖣠" # Injecting an unmapped character
                    
                tokens = self.segmenter.tokenize(str(v))
                ids = self.vocab.encode(tokens)
                lang_id = (daughter_cols.index(lang) + 1)
                
                # Add <SEP> between different cognates
                if len(src_tokens_raw) > 0:
                    src_tokens_raw.append(sep_id)
                    src_langs_raw.append(lang_id)
                    labels.append(f"{lang}:<SEP>")
                
                print(f"Lang: {lang:<12} (ID: {lang_id}) | Tokens: {tokens} -> IDs: {ids}")
                
                src_tokens_raw.extend(ids)
                src_langs_raw.extend([lang_id] * len(tokens))
                # Map token string to label (handling unknown characters)
                mapped_tokens = [t if t in self.vocab.token_to_idx else "<UNK>" for t in tokens]
                labels.extend([f"{lang}:{t}" for t in mapped_tokens])

        # Add <BOS> and <EOS> just like JambuCognateDataset does
        first_lang = src_langs_raw[0] if src_langs_raw else 0
        last_lang = src_langs_raw[-1] if src_langs_raw else 0
        
        src_tokens_raw = [bos_id] + src_tokens_raw + [eos_id]
        src_langs_raw = [first_lang] + src_langs_raw + [last_lang]
        labels = [f"BOS"] + labels + [f"EOS"]
        
        # Simulate DataLoader Padding (add 3 PAD tokens)
        src_tokens_raw.extend([pad_id, pad_id, pad_id])
        src_langs_raw.extend([0, 0, 0])  # Padding gets Language ID 0
        labels.extend(["PAD", "PAD", "PAD"])

        # 3. Feature Matrix Trace
        print("\n[STEP 3] PHONOLOGICAL FEATURE LOOKUP (Special Tokens)")
        special_indices = [0, -4, -1] # BOS, EOS, PAD
        try:
            sep_idx = labels.index("prakrit:<SEP>")
            special_indices.insert(1, sep_idx)
        except ValueError:
            pass
            
        for tid_idx in special_indices:
            tid = src_tokens_raw[tid_idx]
            token_str = labels[tid_idx]
            vector = self.feature_matrix[tid]
            print(f"Token: {token_str:<12} (ID: {tid:<2}) | Vector (last 5 dims): {vector[-5:].tolist()}")

        # 4. Mask Logic Trace
        print("\n[STEP 4] MASKING LOGIC VERIFICATION")
        device = next(self.model.parameters()).device
        t_langs = torch.tensor([src_langs_raw], dtype=torch.long).to(device)
        
        # Exact block mask logic from fevet_ia.py
        block_mask = (t_langs.unsqueeze(-1) != t_langs.unsqueeze(1))
        
        # FIX: Ensure padding queries can attend to valid tokens to prevent NaN outputs
        block_mask = block_mask & (t_langs.unsqueeze(-1) != 0)
        
        nhead = self.underlying_model.nhead
        seq_len = t_langs.size(1)
        block_mask_expanded = block_mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(1 * nhead, seq_len, seq_len)

        self.print_mask_grid(block_mask_expanded, labels, "ENCODER BLOCK-DIAGONAL MASK")

        # 5. Causal Mask Trace
        print("\n[STEP 5] DECODER CAUSAL MASK")
        tgt_len = 6
        causal_mask = get_causal_mask(tgt_len, device)
        causal_labels = ["<BOS>", "t", "a", "r", "g", "<EOS>"]
        self.print_mask_grid(causal_mask, causal_labels, "DECODER CAUSAL MASK")

def main():
    # --- Configuration ---
    MODEL_PATH = "best_fevet_ia_model.pt" 
    DAUGHTER_COLS = [
        "pali", "prakrit", "panjabi", "sindhi", "romani", "kashmiri", "shina",
        "hindi", "garhwali", "nepali", "bhojpuri", "maithili", "bengali", 
        "oriya", "assamese", "gujarati", "marathi", "konkani", "sinhala", "dhivehi"
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing diagnostic environment...")
    
    # 1. Rebuild data 
    forms_raw, langs_raw, params_raw = download_jambu_data()
    segmenter = LinguisticSegmenter()
    df = process_jambu_to_dataframe_unified(forms_raw, langs_raw, params_raw, DAUGHTER_COLS)
    
    # 1.5 EXACT SPLIT REPLICATION (This guarantees Vocab Size is exactly 311)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=67)
    train_idx, _ = next(gss.split(df, groups=df['Parameter_ID']))
    train_df = df.iloc[train_idx]
    
    # Build Vocab using ONLY the training split, exactly as in fevet_ia.py
    all_tokens = []
    for col in ['sanskrit'] + DAUGHTER_COLS:
        for val in train_df[col].dropna():
            if isinstance(val, list):
                for v in val: 
                    if str(v) != "-":
                        all_tokens.append(segmenter.tokenize(str(v)))
            else:
                if str(val) != "-":
                    all_tokens.append(segmenter.tokenize(str(val)))
    
    vocab = ReconstructionVocab()
    vocab.build_vocab(all_tokens, [])
    print(f"Reconstructed Vocab Size: {len(vocab)} (Should be 311)")
    feature_matrix = build_phonological_matrix(vocab).to(device)

    # 2. Initialize Model
    model = FeVeTTransformer(
        vocab_size=len(vocab),
        num_features=34,
        feature_matrix=feature_matrix,
        num_languages=len(DAUGHTER_COLS),
        d_model=128,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=2,
        dim_feedforward=256,
    ).to(device)

    # 3. Load Weights
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        raw_state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        
        # Handle nested checkpoints (e.g., from latest_checkpoint.pt)
        if 'model_state_dict' in raw_state_dict:
            raw_state_dict = raw_state_dict['model_state_dict']
            
        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            clean_key = k.replace("_orig_mod.", "")
            clean_state_dict[clean_key] = v
            
        model.load_state_dict(clean_state_dict)
        print("Model weights successfully loaded!")
    else:
        print(f"Warning: {MODEL_PATH} not found. Running diagnostics on an untrained model.")

    model.eval()

    # 4. Run Diagnostics (We can run this on the original full 'df' to see an actual row)
    diag = FeVeTDiagnosticSuite(model, vocab, segmenter, feature_matrix)
    diag.run_trace(df, DAUGHTER_COLS)

if __name__ == "__main__":
    main()