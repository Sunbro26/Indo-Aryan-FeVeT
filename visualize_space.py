# =====================================================================
# Standalone 2D & 3D t-SNE Plot Generator (Import-Only Version)
# =====================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from sklearn.manifold import TSNE

# Import your updated main file
import fevet_ia  

# Bind the segmenter inside the fevet_ia namespace to prevent the scoping NameError
fevet_ia.segmenter = fevet_ia.LinguisticSegmenter()

# =====================================================================
# Unified Phonetic Class Mapping & Color Specification [1]
# =====================================================================
def get_color_map():
    return {
        'Oral Vowel': '#aec7e8',         # Light Blue [1]
        'Nasalized Vowel': '#1f77b4',    # Dark Blue [1]
        'Glide': '#bcbd22',              # Yellow-Green [1]
        'Liquid/Rhotic': '#8c564b',      # Brown [1]
        'Liquid/Lateral': '#e377c2',     # Pink [1]
        'Labial Consonant': '#ff7f0e',   # Orange [1]
        'Coronal Consonant': '#2ca02c',  # Green [1]
        'Retroflex Consonant': '#9467bd',# Purple [1]
        'Palatal Consonant': '#17becf',  # Cyan [1]
        'Dorsal Consonant': '#d62728',   # Red [1]
        'Glottal/Fricative': '#7f7f7f',  # Grey [1]
        'Other/Special': '#c7c7c7'       # Light Grey [1]
    }

def get_phonetic_classes(indices_to_plot, labels, features, vocab):
    classes = []
    
    # Base character sets to cleanly resolve rhotics, laterals, and glides
    rhotics_base = {"r", "ɾ", "ʁ", "ɽ"}
    laterals_base = {"l", "ʎ", "ɭ"}
    glides_base = {"w", "v", "y"}
    
    for i, idx in enumerate(indices_to_plot):
        feat = features[idx].numpy()
        token = labels[i]
        
        is_vowel = feat[1] == 1.0
        is_nasal_vow = feat[10] == 1.0
        is_labial = feat[20] == 1.0
        is_coronal = feat[21] == 1.0
        is_retroflex = feat[22] == 1.0
        is_palatal = feat[23] == 1.0
        is_dorsal = feat[24] == 1.0
        is_glottal = feat[25] == 1.0
        
        # Clean the raw token string to safely isolate base chars [1]
        clean_token = token.replace("ʰ", "").replace("ʱ", "").replace(":", "").replace("ː", "")
        nfd_token = unicodedata.normalize('NFD', clean_token)
        base_chars = [c for c in nfd_token if not unicodedata.combining(c)]
        base_char = base_chars[0] if base_chars else ""
        
        # Rigorously isolate Liquids and Glides using base characters to prevent Coronal merging [1]
        if is_vowel:
            if is_nasal_vow: 
                classes.append('Nasalized Vowel')
            else: 
                classes.append('Oral Vowel')
        elif base_char in rhotics_base:
            classes.append('Liquid/Rhotic')
        elif base_char in laterals_base:
            classes.append('Liquid/Lateral')
        elif base_char in glides_base:
            classes.append('Glide')
        elif is_labial: 
            classes.append('Labial Consonant')
        elif is_coronal: 
            classes.append('Coronal Consonant')
        elif is_retroflex: 
            classes.append('Retroflex Consonant')
        elif is_palatal: 
            classes.append('Palatal Consonant')
        elif is_dorsal: 
            classes.append('Dorsal Consonant')
        elif is_glottal: 
            classes.append('Glottal/Fricative')
        else: 
            classes.append('Other/Special')
    return classes

# =====================================================================
# 2D and 3D t-SNE Plotters
# =====================================================================
def visualize_phonetic_space_2d(model, vocab, feature_matrix, device, save_path="phonetic_space_ia_30d_2d.png"):
    model.eval()
    features = feature_matrix.cpu()  # Fix: Rebuilt matrix guarantees up-to-date features [1]
    with torch.no_grad():
        embeddings = model.encoder_proj(features.to(device)).cpu().numpy()
        
    indices_to_plot = []
    labels = []
    for token, idx in vocab.token_to_idx.items():
        if token not in vocab.special_tokens and not (token.startswith("[") and token.endswith("]")):
            feat = features[idx].numpy()
            if feat[1:].sum() == 0:
                continue
            indices_to_plot.append(idx)
            labels.append(token)
            
    embeddings_to_plot = embeddings[indices_to_plot]
    classes = get_phonetic_classes(indices_to_plot, labels, features, vocab)
    color_map = get_color_map()
    
    perplexity_val = min(18, len(labels)-1)
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, max_iter=1500, init='pca')
    reduced = tsne.fit_transform(embeddings_to_plot)
    
    plt.figure(figsize=(11, 9), dpi=150)
    unique_classes = sorted(list(set(classes)))
    for cls in unique_classes:
        cls_indices = [i for i, c in enumerate(classes) if c == cls]
        plt.scatter(reduced[cls_indices, 0], reduced[cls_indices, 1],
                    color=color_map[cls], label=cls, s=90, alpha=0.85, edgecolors='black', linewidths=0.8)
                    
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]), xytext=(4, 4), textcoords='offset points', fontsize=10, fontweight='bold', alpha=0.9)
        
    plt.title("FeVeT Learned 2D Phonological Latent Space", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title="Phonetic Classes", title_fontsize=11, fontsize=10, loc="upper right")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Matplotlib] 2D spatial visualization complete. Image saved to: {save_path}")

def visualize_phonetic_space_3d(model, vocab, feature_matrix, device, save_path="phonetic_space_ia_30d_3d.png"):
    model.eval()
    features = feature_matrix.cpu()  # Fix: Rebuilt matrix guarantees up-to-date features [1]
    with torch.no_grad():
        embeddings = model.encoder_proj(features.to(device)).cpu().numpy()
        
    indices_to_plot = []
    labels = []
    for token, idx in vocab.token_to_idx.items():
        if token not in vocab.special_tokens and not (token.startswith("[") and token.endswith("]")):
            feat = features[idx].numpy()
            if feat[1:].sum() == 0:
                continue
            indices_to_plot.append(idx)
            labels.append(token)
            
    embeddings_to_plot = embeddings[indices_to_plot]
    classes = get_phonetic_classes(indices_to_plot, labels, features, vocab)
    color_map = get_color_map()
    
    perplexity_val = min(18, len(labels)-1)
    tsne = TSNE(n_components=3, perplexity=perplexity_val, random_state=42, max_iter=1500, init='pca')
    reduced = tsne.fit_transform(embeddings_to_plot)
    
    fig = plt.figure(figsize=(11, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1]) 
    
    unique_classes = sorted(list(set(classes)))
    for cls in unique_classes:
        cls_indices = [i for i, c in enumerate(classes) if c == cls]
        ax.scatter(reduced[cls_indices, 0], reduced[cls_indices, 1], reduced[cls_indices, 2],
                   color=color_map[cls], label=cls, s=80, alpha=0.8, edgecolors='black', linewidths=0.5)
                   
    for i, label in enumerate(labels):
        ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], label, fontsize=8, fontweight='bold', alpha=0.8)
        
    ax.set_title("FeVeT Learned 3D Phonological Latent Space", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title="Phonetic Classes", title_fontsize=11, fontsize=9, loc="upper right")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Matplotlib] 3D spatial visualization complete. Image saved to: {save_path}")

# =====================================================================
# Main Execution Block
# =====================================================================
if __name__ == "__main__":
    from sklearn.model_selection import GroupShuffleSplit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing standalone space visualizer on: {device}")

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

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint '{checkpoint_path}'. Train the model first.")

    # 2. Download JAMBU & parse comparative matrices
    print("Downloading comparative lexicon...")
    forms_raw, langs_raw, params_raw = fevet_ia.download_jambu_data()
    df = fevet_ia.process_jambu_to_dataframe_unified(forms_raw, langs_raw, params_raw, daughter_columns)
    df = df.reset_index(drop=True)

    print("Creating dataset and splits to align exact vocabulary mapping...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=67)
    train_idx, val_idx = next(gss.split(df, groups=df['Parameter_ID']))
    train_df = df.iloc[train_idx].reset_index(drop=True)

    # Reconstruct vocabulary strictly from train_df [1]
    print("\nBuilding Reconstruction Vocabulary...")
    train_raw_tokens = []
    for col in [target_column] + daughter_columns:
        for value in train_df[col].dropna():
            if isinstance(value, list):
                # Safely unpack list values
                for v in value:
                    if str(v) != "-":
                        train_raw_tokens.append(fevet_ia.segmenter.tokenize(str(v)))
            else:
                if str(value) != "-":
                    train_raw_tokens.append(fevet_ia.segmenter.tokenize(str(value)))

    vocab = fevet_ia.ReconstructionVocab()
    vocab.build_vocab(train_raw_tokens, [])
    print(f"Total Unique Vocabulary Tokens: {len(vocab)}")

    # 4. Re-build Phonological Matrix
    feature_matrix = fevet_ia.build_phonological_matrix(vocab).to(device)

    # 5. Instantiate model architecture using matched hyperparameters
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

    # 6. Load weights from the checkpoint (with compiled key mitigation)
    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Strip "_orig_mod." prefix from key names if they were compiled during training
    clean_state_dict = {
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v 
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state_dict)

    # 7. Execute 2D and 3D visualization functions [1]
    print("Generating 2D t-SNE visualization...")
    visualize_phonetic_space_2d(
        model, vocab, feature_matrix, device, 
        save_path="phonetic_space_ia_30d_2d.png"
    )
    print("Generating 3D t-SNE visualization...")
    visualize_phonetic_space_3d(
        model, vocab, feature_matrix, device, 
        save_path="phonetic_space_ia_30d_3d.png"
    )