# =====================================================================
# Imports, Auxiliary Classes, and Helper Functions (FeVeT-IA)
# =====================================================================

import io
import re
import math
import os
import random
import urllib.request
import collections
import pandas as pd
import numpy as np
import numba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from sklearn.model_selection import GroupShuffleSplit
import unicodedata
from functools import partial

# =====================================================================
# Numba JIT Kernels (Executed at Compiled C-Speed)
# =====================================================================

@numba.njit(cache=True)
def _edit_distance_kernel(seq1: np.ndarray, seq2: np.ndarray) -> float:
    m, n = len(seq1), len(seq2)
    prev = np.arange(n + 1, dtype=np.float32)
    curr = np.zeros(n + 1, dtype=np.float32)
    
    for i in range(1, m + 1):
        curr[0] = float(i)
        val1 = seq1[i-1]
        for j in range(1, n + 1):
            cost = 0.0 if val1 == seq2[j-1] else 1.0
            curr[j] = min(prev[j] + 1.0, curr[j-1] + 1.0, prev[j-1] + cost)
        prev[:] = curr
    return prev[n]


@numba.njit(cache=True)
def _feature_weighted_edit_distance_kernel(idx1: np.ndarray, idx2: np.ndarray, cost_matrix: np.ndarray) -> float:
    m, n = len(idx1), len(idx2)
    prev = np.arange(n + 1, dtype=np.float32)
    curr = np.zeros(n + 1, dtype=np.float32)
    
    for i in range(1, m + 1):
        curr[0] = float(i)
        id1 = idx1[i-1]
        for j in range(1, n + 1):
            id2 = idx2[j-1]
            # Since the matrix diagonal is 0, matching IDs automatically yield 0.0 cost
            cost = 0.0 if id1 == id2 else float(cost_matrix[id1, id2])
            curr[j] = min(prev[j] + 1.0, curr[j-1] + 1.0, prev[j-1] + cost)
        prev[:] = curr
    return prev[n]

# =====================================================================
# Indo-Aryan Orthography Normalization Helper
# =====================================================================
def normalize_indo_aryan_orthography(text: str, is_oia: bool = False) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.strip().lower()
    
    # 1. Strip Vedic pitch accents 
    accent_map = {
        "á": "a", "í": "i", "ú": "u", "é": "e", "ó": "o",
        "ā́": "ā", "ī́": "ī", "ū́": "ū",
        "à": "a", "ì": "i", "ù": "u", "è": "e", "ò": "o",
        "ḗ": "ē", "ḕ": "ē", "ṓ": "ō", "ṑ": "ō",  "ǟ": "ā"
    }
    for accented, base in accent_map.items():
        text = text.replace(accented, base)
    text = text.replace("\u0301", "").replace("\u0300", "")
    
    # 2. Map vocalic liquid vs retroflex flap
    if is_oia:
        text = text.replace("ṛ", "r̥").replace("ḷ", "l̥")
        text = text.replace("r̩", "r̥")
    else:
        text = text.replace("ṛh", "ɽʱ").replace("ṛʰ", "ɽʱ").replace("ṛ", "ɽ")

    # 3. Standardize Aspiration representations
    aspiration_map = {
      "ph": "pʰ", "bh": "bʱ", "th": "tʰ", "dh": "dʱ",
      "kh": "kʰ", "gh": "gʱ", "ch": "cʰ", "jh": "jʱ",
      "ṭh": "ṭʰ", "ḍh": "ḍʱ", 
      "c̣h": "c̣ʰ"
    }
    for raw_asp, clean_asp in aspiration_map.items():
        text = text.replace(raw_asp, clean_asp)
        
    # Standardize Sonorants ONLY for non-OIA (MIA/NIA) languages
    if not is_oia:
        sonorant_asp_map = {
            "mh": "mʱ", "nh": "nʱ", "lh": "lʱ", "rh": "rʱ", "vh": "vʱ",
            "ṇh": "ṇʱ", "ñh": "ñʱ", "ṅh": "ṅʱ", "yh": "yʱ"
        }
        for raw_asp, clean_asp in sonorant_asp_map.items():
            text = text.replace(raw_asp, clean_asp)
        
    text = text.replace("ã̄", "ā̃")

    superscript_map = {"ⁱ": "i", "ᵃ": "a", "ᵘ": "u", "ᵉ": "e", "ᵒ": "o", "ᵊ": "ə", "ⁿ": "m̐"}
    for sup, base in superscript_map.items():
        text = text.replace(sup, base)

    # 4. Unify conflicting aspiration superscript characters
    for v_cons in ["b", "d", "ḍ", "j", "g", "ɦ", "m", "n", "ṇ", "ñ", "ṅ", "ŋ", "l", "r", "v", "w", "y", "ɽ", "z", "dz"]:
        text = text.replace(v_cons + "ʰ", v_cons + "ʱ")
        
    for vl_cons in ["p", "t", "ṭ", "c", "c̣", "k", "s", "ś", "ṣ", "f", "x", "ts"]:
        text = text.replace(vl_cons + "ʱ", vl_cons + "ʰ")
    
    return text

# =====================================================================
# Automated JAMBU Dataset Downloader and Pivot Parser
# =====================================================================
def download_jambu_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    branches = ["master", "main"]
    forms_df = None
    langs_df = None
    params_df = None
    
    for branch in branches:
        try:
            forms_url = f"https://raw.githubusercontent.com/moli-mandala/data/{branch}/cldf/forms.csv"
            langs_url = f"https://raw.githubusercontent.com/moli-mandala/data/{branch}/cldf/languages.csv"
            params_url = f"https://raw.githubusercontent.com/moli-mandala/data/{branch}/cldf/parameters.csv"
            
            print(f"Attempting to download JAMBU data from branch: {branch}...")
            with urllib.request.urlopen(forms_url) as r:
                forms_df = pd.read_csv(io.StringIO(r.read().decode('utf-8')), low_memory=False)
            with urllib.request.urlopen(langs_url) as r:
                langs_df = pd.read_csv(io.StringIO(r.read().decode('utf-8')), low_memory=False)
            with urllib.request.urlopen(params_url) as r:
                params_df = pd.read_csv(io.StringIO(r.read().decode('utf-8')), low_memory=False)
            print("Successfully retrieved JAMBU tables.")
            break
        except Exception as e:
            continue
            
    if forms_df is None or langs_df is None or params_df is None:
        raise IOError("Could not download Jambu CLDF files from GitHub.")
        
    return forms_df, langs_df, params_df

def process_jambu_to_dataframe_unified(forms_df: pd.DataFrame, langs_df: pd.DataFrame, params_df: pd.DataFrame,
                                       daughter_langs: List[str], filter_tatsamas: bool = True) -> pd.DataFrame:
    
    lang_id_to_name = dict(zip(langs_df['ID'], langs_df['Name'].str.lower()))
    
    def normalize_lang_name(name: str) -> str:
        if not isinstance(name, str): return "unknown"
        name = name.lower().strip()
        
        if "old indo-aryan" in name or "sanskrit" in name or name in ["oia", "skt", "indo-aryan"]: return "sanskrit"
        if "pali" in name: return "pali"
        if "prakrit" in name or "middle indo-aryan" in name or name in ["mia", "pkt"]: return "prakrit"
        if "romani" in name or "gypsy" in name: return "romani"
        if "shina" in name or "ṣiṇ" in name: return "shina"
        if "dhivehi" in name or "maldivian" in name: return "dhivehi"
        if "garhwali" in name: return "garhwali"
        if "bhojpuri" in name: return "bhojpuri"
        if "maithili" in name: return "maithili"
        if "konkani" in name: return "konkani"
        if any(d in name for d in ["panjabi", "punjabi", "lahnda", "lehndi", "dogri", "pahari", "pothohari"]): return "panjabi"
        if "hindi" in name or "urdu" in name or "hindustani" in name or "awadhi" in name: return "hindi"
        if "bengali" in name or "bangla" in name: return "bengali"
        if "sindhi" in name: return "sindhi"
        if "gujarati" in name: return "gujarati"
        if "nepali" in name: return "nepali"
        if "marathi" in name: return "marathi"
        if "oriya" in name or "odia" in name: return "oriya"
        if "assamese" in name or "asamiya" in name: return "assamese"
        if "sinhala" in name or "sinhalese" in name: return "sinhala"
        if "kashmiri" in name or "kaśmīrī" in name: return "kashmiri"
        
        return name

    forms_df['Language_Name'] = forms_df['Language_ID'].map(lang_id_to_name).apply(normalize_lang_name)
    daughter_langs = [l.lower() for l in daughter_langs]
    
    all_langs_to_fetch = daughter_langs + ["sanskrit"]
    filtered_df = forms_df[forms_df['Language_Name'].isin(all_langs_to_fetch)].copy()
    
    def apply_norm(row):
        is_oia = (row['Language_Name'] == 'sanskrit')
        return normalize_indo_aryan_orthography(row['Form'], is_oia=is_oia)
        
    filtered_df['Form'] = filtered_df.apply(apply_norm, axis=1)
    
    # Group into lists of unique dialectal representations per language cell
    grouped = filtered_df.groupby(['Parameter_ID', 'Language_Name'])['Form'].apply(lambda x: list(x.dropna().unique())).unstack('Language_Name')
    
    for col in all_langs_to_fetch:
        if col not in grouped.columns:
            grouped[col] = np.nan
            
    # Fill missing with empty representations
    for col in all_langs_to_fetch:
        grouped[col] = grouped[col].apply(lambda x: x if isinstance(x, list) else ["-"])
        
    headword_col = next((col for col in ['Name', 'Form', 'Value', 'ID'] if col in params_df.columns), None)
    if headword_col is None: raise ValueError("Could not find headword column in parameters.csv.")
    param_map = dict(zip(params_df['ID'], params_df[headword_col]))

    def get_sanskrit_targets(row, pid):
        attested = row.get('sanskrit', ["-"])
        if attested != ["-"] and len(attested) > 0 and attested[0] != "":
            return attested
        fallback = param_map.get(pid, '')
        norm = normalize_indo_aryan_orthography(str(fallback), is_oia=True) if fallback else '-'
        return [norm]

    expanded_rows = []
    filtered_count = 0
    tatsama_clusters = [
        # Original Rhotic Conjuncts
        "tr", "pr", "kr", "gr", "dr", "br", "mr", "śr", "ndr",
        # Original Sibilant + Stop/Nasal Conjuncts
        "st", "sm", "sn", "stʰ", "ṣṭ", "ṣṭʰ", "sp", "spʰ", "śc", "ṣp", "ṣṇ",
        # Original Stop + Stop Conjuncts
        "kt", "pt", "gd", "db", "dg",
        # S-Class Defining Conjuncts
        "kṣ", "jñ",
        # Remaining Rhotic Coda Conjuncts (retained across syllables in Tatsamas)
        "rt", "rtʰ", "rd", "rdʱ", "rg", "rm", "rv", "rj", "rṇ"
    ]
    
    def is_tatsama(val: str, target: str) -> bool:
        return any(cluster in val for cluster in tatsama_clusters)

    # Consolidate all available variations into single-row lists of variants
    for pid, row in grouped.iterrows():
        skt_targets = get_sanskrit_targets(row, pid)
        if skt_targets == ["-"] or skt_targets == [""] or skt_targets == ["nan"]:
            continue
            
        skt_target = skt_targets[0]  # Standardize Sanskrit representation
        lang_lists = {}
        has_valid_daughter = False
        
        for lang in daughter_langs:
            raw_vars = row[lang]
            valid_vars = []
            for v in raw_vars:
                if v == "-": continue
                exempt_langs = ["kashmiri", "shina", "romani", "pali", "sinhala", "dhivehi"]
                if filter_tatsamas and lang not in exempt_langs and is_tatsama(v, skt_target):
                    filtered_count += 1
                    continue
                valid_vars.append(v)
                
            if not valid_vars:
                valid_vars = ["-"]
            else:
                has_valid_daughter = True
                
            lang_lists[lang] = valid_vars
            
        if not has_valid_daughter:
            continue
            
        # Exactly 1 consolidated row per cognate parameter
        new_row = {'Parameter_ID': pid, 'sanskrit': skt_target}
        for lang in daughter_langs:
            new_row[lang] = lang_lists[lang]
        expanded_rows.append(new_row)
            
    pivoted = pd.DataFrame(expanded_rows)
    
    print(f"Linguistic Tatsama Filter masked {filtered_count} modern literary borrowings.")
    print(f"Pivoted Indo-Aryan Comparative Dataset: Consolidated into {len(pivoted)} unique etymon groups.")
    return pivoted

# =====================================================================
# Helpers, Segmenters, and Dataset Definitions
# =====================================================================
def precompute_feature_cost_matrix(feature_matrix: torch.Tensor) -> np.ndarray:
    """Precomputes the cosine distance between all pairs of characters in the vocab."""
    fm_cpu = feature_matrix.cpu()
    sim_matrix = torch.matmul(fm_cpu, fm_cpu.T)
    norms = torch.norm(fm_cpu, p=2, dim=1, keepdim=True)
    denom = torch.matmul(norms, norms.T)
    cos_sim = sim_matrix / torch.clamp(denom, min=1e-8)
    cost_matrix = torch.clamp(1.0 - cos_sim, min=0.0, max=1.0)
    return cost_matrix.numpy()

def make_bool_pad_mask(tokens: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return tokens == pad_idx

def edit_distance(seq1: List[str], seq2: List[str]) -> float:
    # 1. Fast short-circuiting for empty sequences
    if not seq1:
        return float(len(seq2))
    if not seq2:
        return float(len(seq1))
    
    # 2. Map arbitrary strings to temporary local integer IDs
    unique_tokens = list(set(seq1 + seq2))
    token_map = {token: idx for idx, token in enumerate(unique_tokens)}
    
    arr1 = np.array([token_map[t] for t in seq1], dtype=np.int32)
    arr2 = np.array([token_map[t] for t in seq2], dtype=np.int32)
    
    return _edit_distance_kernel(arr1, arr2)

def feature_weighted_edit_distance(seq1: List[str], seq2: List[str], vocab, cost_matrix: np.ndarray) -> float:
    # 1. Fast short-circuiting for empty sequences
    if not seq1:
        return float(len(seq2))
    if not seq2:
        return float(len(seq1))
        
    unk_idx = vocab.token_to_idx.get(vocab.unk_token, 0)
    
    # 2. Extract indices on the Python side to avoid passing the custom 'vocab' object to JIT
    idx1 = np.array([vocab.token_to_idx.get(c, unk_idx) for c in seq1], dtype=np.int32)
    idx2 = np.array([vocab.token_to_idx.get(c, unk_idx) for c in seq2], dtype=np.int32)
    
    return _feature_weighted_edit_distance_kernel(idx1, idx2, cost_matrix)

class LinguisticSegmenter:
    def __init__(self):
        self.segment_pattern = re.compile(
            r'(?:[^\W\d_](?:\u0361|\u035c)[^\W\d_]|[^\W\d_])[\u0300-\u036f\u02b0-\u02ffːˑ:]*',
            re.UNICODE
        )
        
        self.vowels = set(
            list("aeiouāīūṛḷā̃ī̃ū̃ẽõœæɔɛəøēōãĩũẽõαεηωⁱᵘᵉᵒᵴᷴáàéèíìóòúùŏŏ̃ăĕĭŭ"
                 "ιουɪɨʌᶸªᵊ") 
            + ["r̥", "l̥"]
        )
        
        self.digraph_map = {
            # Standard Aspirated Plosives
            "kh": "kʰ", "gh": "gʱ", "ch": "cʰ", "jh": "jʱ",
            "ṭh": "ṭʰ", "ḍh": "ḍʱ", "th": "tʰ", "dh": "dʱ",
            "ph": "pʰ", "bh": "bʱ",
            # Kashmiri / Shina Affricates
            "ts": "ts", "dz": "dz", "tsh": "tsʰ", "dzh": "dzʱ",
            "c̣h": "c̣ʰ",
            # Palatal/Retroflex Sibilants (Crucial for Shina, Kashmiri, and Romani)
            "sh": "š",
            "zh": "ž",
            # Aspirated Sonorants & Flaps (Fail-safe)
            "mh": "mʱ", "nh": "nʱ", "lh": "lʱ", "rh": "rʱ", "vh": "vʱ",
            "ṇh": "ṇʱ", "ñh": "ñʱ", "ṅh": "ṅʱ", "yh": "yʱ",
            "ṛh": "ɽʱ"
        }

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str) or pd.isna(text) or text.strip() == "": 
            return []
            
        cleaned = text.strip()
        if cleaned == "-":
            return ["-"]
            
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        cleaned = re.sub(r'\(.*?\)', '', cleaned)
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = cleaned.replace("*", "").replace("?", "").replace(",", "")
        cleaned = cleaned.replace("-", "").replace(" ", "").replace("°", "")
        cleaned = cleaned.replace("'", "").replace('"', "").replace(";", "")
        
        raw_tokens = self.segment_pattern.findall(cleaned)
        if not raw_tokens: 
            return []
        
        # PASS 1: Greedy grouping AND automatic IPA normalization
        grouped = []
        i = 0
        while i < len(raw_tokens):
            tri = "".join(raw_tokens[i:i+3])
            di = "".join(raw_tokens[i:i+2])
            
            if i < len(raw_tokens) - 2 and tri in self.digraph_map:
                grouped.append(self.digraph_map[tri])
                i += 3
            elif i < len(raw_tokens) - 1 and di in self.digraph_map:
                grouped.append(self.digraph_map[di])
                i += 2
            else:
                grouped.append(raw_tokens[i])
                i += 1
                
        # PASS 2: Group Geminates (identical consonants or aspirated transitions)
        tokens = []
        j = 0
        while j < len(grouped):
            t1 = grouped[j]
            if j < len(grouped) - 1:
                t2 = grouped[j+1]
                
                if t1 == t2 and t1 not in self.vowels:
                    tokens.append(t1 + t2)
                    j += 2
                    continue
                    
                if t1 not in self.vowels and (t2 == t1 + 'h' or t2 == t1 + 'ʰ' or t2 == t1 + 'ʱ'):
                    tokens.append(t1 + t2)
                    j += 2
                    continue
                    
            tokens.append(t1)
            j += 1
            
        return tokens

class ReconstructionVocab:
    def __init__(self, pad_token="<PAD>", unk_token="<UNK>", bos_token="<BOS>", eos_token="<EOS>", sep_token="<SEP>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token, sep_token]
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
    def build_vocab(self, tokenized_corpora: List[List[str]], language_tags: List[str]):
        for s_token in self.special_tokens: self._add_token(s_token)
        for tag in language_tags: self._add_token(tag)
        frequencies = collections.Counter(token for sequence in tokenized_corpora for token in sequence)
        for token, _ in frequencies.most_common(): self._add_token(token)
    def _add_token(self, token: str):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
    def encode(self, tokens: List[str], add_bos=False, add_eos=False) -> List[int]:
        encoded = [self.token_to_idx.get(t, self.token_to_idx[self.unk_token]) for t in tokens]
        if add_bos: encoded.insert(0, self.token_to_idx[self.bos_token])
        if add_eos: encoded.append(self.token_to_idx[self.eos_token])
        return encoded
    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx_to_token.get(i, self.unk_token) for i in indices]
    def __len__(self): return len(self.token_to_idx)

class JambuCognateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str, daughter_cols: List[str], vocab: ReconstructionVocab, segmenter: LinguisticSegmenter):
        self.vocab = vocab
        self.segmenter = segmenter
        self.target_col = target_col
        self.daughter_cols = daughter_cols
        self.lang_to_idx = {lang: i + 1 for i, lang in enumerate(daughter_cols)}
        self.lang_to_idx["TARGET"] = len(daughter_cols) + 1
        self.lang_to_idx["PAD"] = 0
        self.samples = self._process_dataset(df)

    def _process_dataset(self, df: pd.DataFrame) -> List[Tuple[List[int], List[int], List[int], List[int], List[int]]]:
        processed_records = []
        for _, row in df.iterrows():
            src_tokens, src_langs, src_pos = [], [], []
            for daughter in self.daughter_cols:
                vals = row[daughter]
                if not isinstance(vals, list):
                    vals = [vals]
                
                for val in vals:
                    val = str(val).strip()
                    if val == "-" or val == "": 
                        continue
                    
                    lang_id = self.lang_to_idx[daughter]
                    tokens = self.segmenter.tokenize(val)
                    encoded_tokens = self.vocab.encode(tokens)
                    
                    # 1. Append SEP to the END of the current cognate sequence
                    encoded_tokens.append(self.vocab.token_to_idx[self.vocab.sep_token])
                    
                    src_tokens.extend(encoded_tokens)
                    src_langs.extend([lang_id] * len(encoded_tokens))
                    
                    # 2. Assign positions 0 to N. The SEP token cleanly gets position N.
                    src_pos.extend(list(range(len(encoded_tokens))))
            
            if not src_tokens:  
                continue
                
            # 3. NO BOS/EOS added to the source! The paper doesn't use them in the encoder.
            # This fixes the block_mask asymmetry.
            
            tgt_val = str(row[self.target_col]).strip()
            tgt_raw_tokens = self.segmenter.tokenize(tgt_val)
            
            # Target STILL gets BOS/EOS, which is correct for autoregressive decoding
            tgt_tokens = self.vocab.encode(tgt_raw_tokens, add_bos=True, add_eos=True)
            tgt_langs = [self.lang_to_idx["TARGET"]] * len(tgt_tokens)
            
            processed_records.append((src_tokens, src_langs, src_pos, tgt_tokens, tgt_langs))
        return processed_records

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        src_t, src_l, src_p, tgt_t, tgt_l = self.samples[idx]
        return (torch.tensor(src_t, dtype=torch.long), 
                torch.tensor(src_l, dtype=torch.long),
                torch.tensor(src_p, dtype=torch.long),
                torch.tensor(tgt_t, dtype=torch.long), 
                torch.tensor(tgt_l, dtype=torch.long))


def collate_fn(batch, pad_idx: int, lang_dropout_prob: float = 0.0):
    src_tokens, src_langs, src_pos, tgt_tokens, tgt_langs = [], [], [], [], []
    
    for src_t, src_l, src_p, tgt_t, tgt_l in batch:
        src_t = src_t.clone()
        src_l = src_l.clone()
        src_p = src_p.clone()
        
        if lang_dropout_prob > 0.0:
            unique_langs = torch.unique(src_l)
            for lang_id in unique_langs:
                if lang_id.item() != 0 and random.random() < lang_dropout_prob:
                    mask = (src_l == lang_id)
                    src_t[mask] = pad_idx 
                    src_l[mask] = 0  
                    src_p[mask] = 0  # Align dropped token positions to 0 as well
                    
        src_tokens.append(src_t)
        src_langs.append(src_l)
        src_pos.append(src_p)
        tgt_tokens.append(tgt_t)
        tgt_langs.append(tgt_l)
        
    src_tokens_padded = torch.nn.utils.rnn.pad_sequence(src_tokens, batch_first=True, padding_value=pad_idx)
    src_langs_padded = torch.nn.utils.rnn.pad_sequence(src_langs, batch_first=True, padding_value=0)
    src_pos_padded = torch.nn.utils.rnn.pad_sequence(src_pos, batch_first=True, padding_value=0)
    tgt_tokens_padded = torch.nn.utils.rnn.pad_sequence(tgt_tokens, batch_first=True, padding_value=pad_idx)
    tgt_langs_padded = torch.nn.utils.rnn.pad_sequence(tgt_langs, batch_first=True, padding_value=0)
    
    return src_tokens_padded, src_langs_padded, src_pos_padded, tgt_tokens_padded, tgt_langs_padded


# =====================================================================
# 30-Dimensional Phonological Matrix Builder
# =====================================================================
FEATURE_DIM = 34
FEATURE_NAMES = [
    "is_special", "is_vowel", "is_consonant", "vowel_high", "vowel_mid", "vowel_low",
    "vowel_front", "vowel_central", "vowel_back", "vowel_long", "vowel_nasalized", "consonant_nasal",
    "voiced", "voiceless", "is_retroflex", "is_aspirated", "is_breathy", "is_geminated",
    "is_vocalic_liquid", "is_sibilant", "place_labial", "place_dental", "place_retroflex",
    "place_palatal", "place_velar", "place_glottal", "manner_stop", "manner_fricative",
    "manner_affricate", "manner_liquid_glide", "is_bos", "is_eos", "is_pad", "is_sep"
]

def build_phonological_matrix(vocab) -> torch.Tensor:
    matrix = torch.zeros(len(vocab), FEATURE_DIM)
    
    # Master Vowel Set: Added Greek bases and unusual IPA symbols
    vowels = {"a", "e", "i", "o", "u", "ā", "ī", "ū", "ā̃", "ī̃", "ū̃", "ẽ", "õ", "œ", "æ", "ɔ", "ɛ", "ə", "ǝ", "l̥", "r̥", "ă", "ĕ", "ĭ", "ŏ", "ŭ", "ŏ̃", "á", "à", "é", "è", "í", "ì", "ó", "ò", "ú", "ù", "ⁱ", "ᵘ", "ᵉ", "ᵒ", "ᵴ", "ᷴ", "\u0325", "α", "ε", "ι", "ο", "υ", "ω", "η", "ɪ", "ɨ", "ʌ", "ᶸ", "ª", "ᵊ", "а", "о"}
    
    # Base Height/Place Sets: Added missing IPA bases
    high_v = {"i", "u", "ī", "ū", "í", "ú", "ĩ", "ũ", "ĩ̃", "ū̃", "ü", "ï", "ĭ", "ŭ", "ᶸ", "η", "ⁱ", "ᵘ", "í", "ì", "ú", "ù", "\u0325", "r̥", "l̥", "ɪ", "ɨ"}
    mid_v = {"e", "o", "ẽ", "õ", "ē", "ō", "œ", "ø", "ɔ", "ɛ", "ε", "ö", "ë", "ĕ", "ŏ", "ə", "ǝ", "ᵊ", "ᵒ", "ω", "η", "ᵉ", "ᵒ", "é", "è", "ó", "ò", "ŏ̃", "о"}
    low_v = {"a", "ā", "ā̃", "ã", "æ", "ä", "ă", "ʌ", "ȧ", "ª", "α", "á", "à", "а"}
    
    front_v = {"i", "e", "ẽ", "œ", "æ", "ē", "ĩ", "ī", "ī̃", "í", "ü", "ö", "ï", "ë", "ɛ", "ε", "ĕ", "ĭ", "η", "ⁱ", "ᵉ", "í", "ì", "é", "è", "ī́", "ɪ"}
    central_v = {"a", "ā", "ā̃", "ã", "ə", "ǝ", "ä", "ă", "ʌ", "ȧ", "ᵊ", "ª", "α", "ᵴ", "ᷴ", "á", "à", "\u0325", "r̥", "l̥", "ɨ", "а"}
    back_v = {"u", "o", "õ", "ũ", "ū", "ū̃", "ú", "ɔ", "ŏ", "ŭ", "ᶸ", "ᵒ", "ω", "ᵘ", "ᵒ", "ú", "ù", "ó", "ò", "ū́", "ŏ̃","о"}
    
    # Nasal Consonants: Added ň, ǹ, and underdot ṃ
    nasal_consonants = {"m", "n", "ɲ", "ñ", "ŋ", "ɳ", "ṁ", "m̐", "ṃ", "ṇ", "ṅ", "ᵐ", "ṉ", "μ", "ν", "ň", "ǹ", "ṃ"}
    
    # Rhotics & Laterals: Added Greek ρ and λ, voiceless ɬ
    rhotics = {"r", "ɾ", "ʁ", "ɽ", "ṛ", "ṟ", "ρ"}
    laterals = {"l", "ʎ", "ɭ", "ḷ", "ḻ", "λ", "ɬ"}
    glides = {"w", "v", "y", "ṽ", "ỹ"}
    
    # Voiced String: Added ζ, ν, λ, ρ, ᶑ
    voiced = set("bdgvlrmnwβðɣʒʁɲzźžżẓɖɳɽɭɦyḍṇjɟj̈ʣʤᶻɓɗʄɠṟḻδγμζνλρᶑǰ") | nasal_consonants | rhotics | laterals
    
    # Voiceless Set: Added κ, π, ċ, ɬ, ɸ, ḣ
    voiceless = {"p", "t", "k", "f", "s", "h", "c", "q", "x", "ç", "ʃ", "θ", "χ", "ʂ", "ʈ", "ṣ", "ṭ", "ś", "ʦ", "ʧ", "c̣", "č", "ḥ", "ṯ", "ϕ", "τ", "σ", "κ", "π", "ċ", "ɬ", "ɸ", "ḣ", "š", "ʔ", "ʰ"}
    
    # Stops: Added Greek κ, π, β, δ, γ, and retroflex implosive ᶑ, ḏ, ḡ
    stops = {"p", "t", "k", "b", "g", "ʈ", "ɖ", "q", "ṭ", "ḍ", "ɓ", "ɗ", "ʄ", "ɠ", "ṯ", "τ", "κ", "π", "β", "δ", "γ", "ᶑ", "ḏ", "ḡ", "ʔ"}
    
    # Fricatives: Added Greek ζ, ɬ, ɸ, ḣ
    fricatives = {"f", "v", "θ", "ð", "s", "z", "ź", "ž", "ż", "ẓ", "ẓ", "ʃ", "ʒ", "ś", "ṣ", "ʂ", "x", "ɣ", "χ", "ʁ", "h", "ɦ", "β", "ç", "ᶻ", "ḥ", "ϕ", "δ", "γ", "σ", "ζ", "ɬ", "ɸ", "ḣ", "š", "ʰ"}
    affricates = {"t͡ʃ", "t͡s", "d͡ʒ", "d͡z", "c", "j", "ɟ", "ʦ", "ʣ", "ʧ", "ʤ", "c̣", "č", "j̈", "ts", "dz", "tsh", "dzh", "ċ", "ǰ"} 
    
    # Anatomy Places: Added missing consonant matches
    labials = {"p", "b", "m", "f", "v", "w", "β", "m̐", "ṁ", "ᵐ", "ɓ", "ϕ", "μ", "ʷ", "\u02b7", "ṽ", "π", "ɸ", "ṃ"}
    dentals = {"t", "d", "s", "z", "n", "l", "r", "θ", "ð", "ʦ", "ʣ", "ᶻ", "ṯ", "ṉ", "τ", "δ", "σ", "ζ", "λ", "ρ", "ν", "ż", "ǹ", "ɬ", "ḏ"}
    retroflexes = {"ʈ", "ɖ", "ɳ", "ɽ", "ɭ", "ʂ", "ṭ", "ḍ", "ṇ", "ṣ", "ṛ", "ẓ", "ḷ", "c̣", "ɗ", "ḻ", "ᶑ"}
    palatals = {"c", "j", "ñ", "ś", "y", "ɲ", "ç", "ɟ", "č", "j̈", "ʄ", "ʃ", "ʒ", "t͡ʃ", "d͡ʒ", "ʸ", "\u02b2", "ỹ", "ċ", "ň", "ź", "ǵ", "š", "ǰ", "ž"}
    velars = {"k", "g", "ŋ", "x", "ɣ", "χ", "q", "g", "ʁ", "ṅ", "ɠ", "γ", "κ", "ḡ"}
    glottals = {"h", "ɦ", "ʔ", "ḥ", "ḣ", "ʰ"}
    
    for token, idx in vocab.token_to_idx.items():
        if token == vocab.bos_token:
            matrix[idx, 30] = 1.0
            continue
        if token == vocab.eos_token:
            matrix[idx, 31] = 1.0
            continue
        if token == vocab.pad_token:
            matrix[idx, 32] = 1.0
            continue
        if token == vocab.sep_token:
            matrix[idx, 33] = 1.0
            continue
        if token == vocab.unk_token or token == "-":
            matrix[idx, 0] = 1.0 # is_special
            continue
            
        normalized_token = unicodedata.normalize('NFD', token)
        
        preserve_marks = {
            '\u0323', '\u0303', '\u0308', '\u0325', '\u0329', '\u0304', 
            '\u0320', '\u0331', '\u0301', '\u0300', '\u030C', '\u0307',
            '\u0361', '\u035C', '\u0306', '\u0310'
        }
        
        clean_chars = []
        for c in normalized_token:
            if unicodedata.combining(c):
                if c in preserve_marks:
                    clean_chars.append(c)
                continue
            clean_chars.append(c)
            
        base_char = unicodedata.normalize('NFC', "".join(clean_chars))
        clean_char = base_char

        is_long = 0.0
        is_gem = 0.0
        
        if any(mark in normalized_token for mark in [":", "ː", "̄", "\u0304"]):
            is_long = 1.0
            
        bases = [c for c in normalized_token if not unicodedata.combining(c)]
        if len(bases) >= 2 and bases[0] == bases[1] and bases[0] not in vowels:
            is_gem = 1.0
            idx_second = clean_char.find(bases[0], 1)
            if idx_second != -1:
                clean_char = clean_char[:idx_second]

        if clean_char not in ["ʰ", "ʱ"]:
            clean_char = clean_char.replace("ʰ", "").replace("ʱ", "")
        clean_char = clean_char.replace(":", "").replace("ː", "")
        
        if len(clean_char) > 1 and clean_char.endswith("h"):
            clean_char = clean_char[:-1]

        # Strip spacing acute, grave, and spacing carons (U+02C7) to prevent matching breaks
        clean_char = clean_char.replace("´", "").replace("ˊ", "").replace("ʹ", "").replace("ˈ", "")
        clean_char = clean_char.replace("ˇ", "").replace("`", "").replace("ˋ", "")
        
        clean_char = clean_char.replace("~", "\u0303").replace("˜", "\u0303")

        # Decompose clean_char to NFD to easily isolate combining nasal marks
        decomposed_clean = unicodedata.normalize('NFD', clean_char)
        
        # Remove combining and spacing nasalization markers
        for nasal_marker in ["\u0303", "\u0310", "~", "˜"]:
            decomposed_clean = decomposed_clean.replace(nasal_marker, "")
            
        # Recompose back to NFC (this leaves clean_char as a pure oral base vowel)
        clean_char = unicodedata.normalize('NFC', decomposed_clean)

        # Check vowels against the fully decomposed NFD string to support double-accented variants
        is_vow = 1.0 if (clean_char in vowels or any(c in vowels for c in normalized_token)) else 0.0
        rep_char = clean_char
        
        base_consonant = bases[0] if bases else ""
        is_cons = 1.0 if ((base_consonant in voiced or base_consonant in voiceless) and not is_vow) else 0.0
        
        place_char = rep_char
        if len(rep_char) >= 3 and (rep_char[1] == '\u0361' or rep_char[1] == '\u035C'):
            place_char = rep_char[2:]

        matrix[idx, 1] = is_vow
        matrix[idx, 2] = is_cons
        
        # Check parent classes using both the precomposed base and constituent NFD characters
        matrix[idx, 3] = 1.0 if (rep_char in high_v or any(c in high_v for c in normalized_token)) else 0.0
        matrix[idx, 4] = 1.0 if (rep_char in mid_v or any(c in mid_v for c in normalized_token)) else 0.0
        matrix[idx, 5] = 1.0 if (rep_char in low_v or any(c in low_v for c in normalized_token)) else 0.0
        matrix[idx, 6] = 1.0 if (rep_char in front_v or any(c in front_v for c in normalized_token)) else 0.0
        matrix[idx, 7] = 1.0 if (rep_char in central_v or any(c in central_v for c in normalized_token)) else 0.0
        matrix[idx, 8] = 1.0 if (rep_char in back_v or any(c in back_v for c in normalized_token)) else 0.0
        matrix[idx, 9] = is_long
        
        # matrix[idx, 10] = 1.0 if (is_vow == 1.0 and "\u0303" in normalized_token) else 0.0
        matrix[idx, 10] = 1.0 if (is_vow == 1.0 and any(m in normalized_token for m in ["\u0303", "\u0310", "~", "˜"])) else 0.0
        # Nasalized consonants (like ṽ, ỹ) are correctly mapped as nasals
        # matrix[idx, 11] = 1.0 if (is_cons == 1.0 and (rep_char in nasal_consonants or any(c in nasal_consonants for c in rep_char) or "\u0303" in normalized_token)) else 0.0
        matrix[idx, 11] = 1.0 if (is_cons == 1.0 and (rep_char in nasal_consonants or any(c in nasal_consonants for c in rep_char) or any(m in normalized_token for m in ["\u0303", "\u0310", "~", "˜"]))) else 0.0
        
        # Voicing and Voiceless features are mapped strictly using the clean base consonant to bypass NFC bugs
        matrix[idx, 12] = 1.0 if (is_cons == 1.0 and base_consonant in voiced) else 0.0
        matrix[idx, 13] = 1.0 if (is_cons == 1.0 and base_consonant in voiceless) else 0.0
        
        is_ret = 1.0 if (is_cons == 1.0 and (rep_char in retroflexes or any(c in retroflexes for c in rep_char))) else 0.0
        matrix[idx, 14] = is_ret
        
        is_asp = 0.0
        aspirate_markers = ["ʰ", "ʱ", "ph", "th", "ṭh", "ch", "kh", "bh", "dh", "ḍh", "gh", "jh"]
        if any(asp in normalized_token for asp in aspirate_markers) or ("h" in token and len(token) > 1):
            is_asp = 1.0
        matrix[idx, 15] = is_asp

        is_breath = 0.0
        if (is_asp == 1.0 and (rep_char in voiced or any(c in voiced for c in rep_char))) or any(c in "ɦʱ" for c in normalized_token):
            is_breath = 1.0
        matrix[idx, 16] = is_breath 
        
        if is_gem == 1.0:
            matrix[idx, 17] = 1.0
            
        matrix[idx, 18] = 1.0 if (base_consonant in rhotics or base_consonant in laterals or "\u0325" in normalized_token) else 0.0
        matrix[idx, 19] = 1.0 if (rep_char in fricatives and (rep_char in "sśṣšʃʒzźžẓʂσ" or any(c in "sśṣšʃʒzźžẓʂσ" for c in rep_char))) else 0.0
        matrix[idx, 20] = 1.0 if (is_cons == 1.0 and is_ret == 0.0 and (place_char in labials or any(c in labials for c in place_char))) else 0.0
        matrix[idx, 21] = 1.0 if (is_cons == 1.0 and is_ret == 0.0 and (place_char in dentals or any(c in dentals for c in place_char))) else 0.0
        matrix[idx, 22] = is_ret
        matrix[idx, 23] = 1.0 if (is_cons == 1.0 and is_ret == 0.0 and (place_char in palatals or any(c in palatals for c in place_char))) else 0.0
        matrix[idx, 24] = 1.0 if (is_cons == 1.0 and (place_char in velars or any(c in velars for c in place_char))) else 0.0
        matrix[idx, 25] = 1.0 if (is_cons == 1.0 and (place_char in glottals or any(c in glottals for c in place_char))) else 0.0

        is_aff = 1.0 if (rep_char in affricates or any(c in affricates for c in rep_char)) else 0.0
        matrix[idx, 26] = 1.0 if (is_aff == 0.0 and (rep_char in stops or any(c in stops for c in rep_char))) else 0.0
        is_liq_glide = 1.0 if (any(c in (laterals | rhotics | glides) for c in rep_char) or "\u0325" in normalized_token) else 0.0
        matrix[idx, 29] = is_liq_glide

        matrix[idx, 27] = 1.0 if (is_aff == 0.0 and is_liq_glide == 0.0 and (rep_char in fricatives or any(c in fricatives for c in rep_char))) else 0.0
        matrix[idx, 28] = is_aff

        if matrix[idx, 1] == 0 and matrix[idx, 2] == 0:
            cat = unicodedata.category(rep_char[0]) if rep_char else ""
            if cat in ["Ll", "Lm"]: 
                matrix[idx, 2] = 1.0
                matrix[idx, 13] = 1.0
    return matrix

# =====================================================================
# Neural Architecture Components
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Remove the unsqueeze so we can index it directly
        self.register_buffer('pe', pe) 
        
    def forward(self, x: torch.Tensor, pos_indices: torch.Tensor = None) -> torch.Tensor:
        if pos_indices is not None:
            # Apply specific local positions (for the Encoder)
            return x + self.pe[pos_indices]
        else:
            # Standard absolute position (for the Decoder)
            return x + self.pe[:x.size(1), :].unsqueeze(0)

class FeVeTTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_features: int, feature_matrix: torch.Tensor,
                 num_languages: int, d_model: int = 128, nhead: int = 4, 
                 num_encoder_layers: int = 4, num_decoder_layers: int = 4, dim_feedforward: int = 512):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.feature_lookup = nn.Parameter(feature_matrix, requires_grad=False)
        self.encoder_proj = nn.Linear(num_features, d_model)
        self.decoder_proj = nn.Linear(num_features, d_model)
        self.lang_embeddings = nn.Embedding(num_languages + 2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.20
        )
        self.char_head = nn.Linear(d_model, vocab_size)
        self.feature_head = nn.Linear(d_model, num_features)
        
    def forward(self, src_tokens, src_langs, src_pos, tgt_tokens, tgt_langs, src_pad_mask=None, tgt_mask=None, tgt_pad_mask=None):
        src_feats = self.feature_lookup[src_tokens]
        tgt_feats = self.feature_lookup[tgt_tokens]
        src_emb = (self.encoder_proj(src_feats) * math.sqrt(self.d_model)) + self.lang_embeddings(src_langs)
        tgt_emb = (self.decoder_proj(tgt_feats) * math.sqrt(self.d_model)) + self.lang_embeddings(tgt_langs)
        src_emb = self.encoder_norm(self.pos_encoder(src_emb, pos_indices=src_pos))
        tgt_emb = self.decoder_norm(self.pos_encoder(tgt_emb, pos_indices=None))

        # Create a mask to prevent cross-cognate attention in the encoder
        batch_size, seq_len = src_langs.shape
        block_mask = (src_langs.unsqueeze(-1) != src_langs.unsqueeze(1))
        
        # FIX: Ensure padding queries can attend to valid tokens to prevent NaN outputs!
        block_mask = block_mask & (src_langs.unsqueeze(-1) != 0)
        
        block_mask = block_mask.unsqueeze(1).expand(-1, self.nhead, -1, -1).reshape(batch_size * self.nhead, seq_len, seq_len)

        # Pass it to the transformer
        dec_out = self.transformer(
            src_emb, tgt_emb, 
            src_mask=block_mask,  # <--- APPLY IT HERE
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_pad_mask, 
            tgt_key_padding_mask=tgt_pad_mask, 
            memory_key_padding_mask=src_pad_mask
        )
        return self.char_head(dec_out), self.feature_head(dec_out)
    
    # Add these to FeVeTTransformer
    def encode(self, src_tokens, src_langs, src_pos, src_pad_mask=None, src_mask=None):
        src_feats = self.feature_lookup[src_tokens]
        src_emb = (self.encoder_proj(src_feats) * math.sqrt(self.d_model)) + self.lang_embeddings(src_langs)
        src_emb = self.encoder_norm(self.pos_encoder(src_emb, pos_indices=src_pos))
        memory = self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_pad_mask)
        return memory

    def decode(self, memory, tgt_tokens, tgt_langs, src_pad_mask=None, tgt_mask=None, tgt_pad_mask=None):
        tgt_feats = self.feature_lookup[tgt_tokens]
        tgt_emb = (self.decoder_proj(tgt_feats) * math.sqrt(self.d_model)) + self.lang_embeddings(tgt_langs)
        tgt_emb = self.decoder_norm(self.pos_encoder(tgt_emb, pos_indices=None))
        dec_out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        return self.char_head(dec_out), self.feature_head(dec_out)

class FeVeTMultiTaskLoss(nn.Module):
    def __init__(self, feature_matrix: torch.Tensor, alpha: float = 0.5, pad_idx: int = 0, ema_momentum: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.pad_idx = pad_idx
        self.feature_lookup = nn.Parameter(feature_matrix, requires_grad=False)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Buffers to track the moving average of loss magnitudes
        self.register_buffer('ema_loss_char', torch.tensor(1.0))
        self.register_buffer('ema_loss_feat', torch.tensor(1.0))
        self.register_buffer('step_count', torch.tensor(0))
        self.ema_momentum = ema_momentum

    def forward(self, char_logits, feature_preds, targets):
        flat_logits = char_logits.reshape(-1, char_logits.size(-1))
        flat_targets = targets.reshape(-1)
        loss_char = self.ce_loss(flat_logits, flat_targets)
        
        gold_features = self.feature_lookup[targets]
        mask = (targets != self.pad_idx).float().unsqueeze(-1)
        raw_bce = self.bce_loss(feature_preds, gold_features)
        loss_feat_raw = (raw_bce * mask).sum() / (mask.sum() * gold_features.size(-1) + 1e-8)
        
        # --- DYNAMIC WEIGHT AVERAGING (Magnitude Balancing) ---
        if self.training:
            if self.step_count.item() == 0:
                self.ema_loss_char.fill_(loss_char.item())
                self.ema_loss_feat.fill_(loss_feat_raw.item())
            else:
                self.ema_loss_char.data = self.ema_momentum * self.ema_loss_char + (1 - self.ema_momentum) * loss_char.detach()
                self.ema_loss_feat.data = self.ema_momentum * self.ema_loss_feat + (1 - self.ema_momentum) * loss_feat_raw.detach()
            self.step_count.add_(1)
            
        # Calculate dynamic scale: What number multiplies feature_loss to equal char_loss?
        # We detach() it so the neural network doesn't try to cheat by manipulating the scale via gradients
        dynamic_scale = (self.ema_loss_char / (self.ema_loss_feat + 1e-8)).detach()
        loss_feat_scaled = loss_feat_raw * dynamic_scale  
        total_loss = (self.alpha * loss_char) + ((1 - self.alpha) * loss_feat_scaled)
        return total_loss, loss_char, loss_feat_raw

# =====================================================================
# Decoders & Evaluation Functions
# =====================================================================
# OPTIMIZATION: Global cache for Transformer causal masks to avoid GPU synchronization overheads.
_MASK_CACHE = {}

def get_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    if sz not in _MASK_CACHE:
        # Added dtype=torch.bool to match your boolean pad masks
        _MASK_CACHE[sz] = torch.nn.Transformer.generate_square_subsequent_mask(
            sz, device=device, dtype=torch.bool
        )
    return _MASK_CACHE[sz]

@torch.no_grad()
def greedy_decode(model: nn.Module, src_tokens: torch.Tensor, src_langs: torch.Tensor, src_pos: torch.Tensor, vocab, target_lang_id: int, max_len: int = 20) -> torch.Tensor:
    model.eval()
    batch_size, device = src_tokens.size(0), src_tokens.device
    bos_idx, eos_idx, pad_idx = vocab.token_to_idx[vocab.bos_token], vocab.token_to_idx[vocab.eos_token], vocab.token_to_idx[vocab.pad_token]
    
    # 1. Safe resolution of compiled models to access custom methods and properties
    underlying_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    nhead = getattr(underlying_model, 'nhead', 4)
    
    tgt_tokens = torch.full((batch_size, max_len + 1), pad_idx, dtype=torch.long, device=device)
    tgt_tokens[:, 0] = bos_idx
    tgt_langs = torch.full((batch_size, max_len + 1), target_lang_id, dtype=torch.long, device=device)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    src_pad_mask = make_bool_pad_mask(src_tokens, pad_idx)
    full_mask = get_causal_mask(max_len + 1, device)
    
    # 2. Build the block-diagonal mask for the encoder once
    seq_len = src_langs.size(1)
    block_mask = (src_langs.unsqueeze(-1) != src_langs.unsqueeze(1))
    
    # FIX: Unmask padding queries
    block_mask = block_mask & (src_langs.unsqueeze(-1) != 0)
    
    block_mask = block_mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(batch_size * nhead, seq_len, seq_len)
    
    # 3. Encode the source sequence ONCE before entering the loop
    memory = underlying_model.encode(
        src_tokens=src_tokens,
        src_langs=src_langs,
        src_pos=src_pos,
        src_pad_mask=src_pad_mask,
        src_mask=block_mask
    )
    
    # 4. Autoregressive generation loop
    for step in range(max_len):
        sz = step + 1
        current_tgt = tgt_tokens[:, :sz]
        current_lng = tgt_langs[:, :sz]
        tgt_mask = full_mask[:sz, :sz]
        
        # Call only the decoder, querying the pre-computed isolated memory
        char_logits, _ = underlying_model.decode(
            memory=memory,
            tgt_tokens=current_tgt,
            tgt_langs=current_lng,
            src_pad_mask=src_pad_mask,
            tgt_mask=tgt_mask,
            tgt_pad_mask=None
        )
        
        next_tokens = torch.argmax(char_logits[:, -1, :], dim=-1)
        next_tokens = torch.where(finished, pad_idx, next_tokens)
        
        tgt_tokens[:, step + 1] = next_tokens
        finished = finished | (next_tokens == eos_idx)
        
        if finished.all(): 
            return tgt_tokens[:, :step + 2]
            
    return tgt_tokens


@torch.no_grad()
def beam_search_decode(model: nn.Module, src_token: torch.Tensor, src_lang: torch.Tensor, src_pos: torch.Tensor,
                       vocab: ReconstructionVocab, target_lang_id: int, 
                       beam_width: int = 10, max_len: int = 20) -> List[Tuple[List[str], float]]:
    model.eval()
    device = src_token.device
    bos_idx, eos_idx, pad_idx = vocab.token_to_idx[vocab.bos_token], vocab.token_to_idx[vocab.eos_token], vocab.token_to_idx[vocab.pad_token]
    
    # 1. Resolve compiled wrapper and gather hyperparameters
    underlying_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    nhead = getattr(underlying_model, 'nhead', 4)
    
    initial_seq = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
    initial_lang = torch.full((1, 1), target_lang_id, dtype=torch.long, device=device)
    beams = [(initial_seq, initial_lang, 0.0)]
    src_pad_mask = make_bool_pad_mask(src_token, pad_idx)
    
    # 2. Build block mask to prevent cross-lingual encoder leakage
    batch_size = src_lang.size(0)  # <-- FIX: Dynamically define batch_size
    seq_len = src_lang.size(1)
    block_mask = (src_lang.unsqueeze(-1) != src_lang.unsqueeze(1))
    
    # FIX: Unmask padding queries (Note: Variable here is named src_lang, not src_langs)
    block_mask = block_mask & (src_lang.unsqueeze(-1) != 0)
    
    block_mask = block_mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(batch_size * nhead, seq_len, seq_len)
    
    # 3. Encode the source ONCE
    memory = underlying_model.encode(
        src_tokens=src_token,
        src_langs=src_lang,
        src_pos=src_pos,
        src_pad_mask=src_pad_mask,
        src_mask=block_mask
    )
    
    for _ in range(max_len):
        candidates = []
        for seq, lang, score in beams:
            if seq[0, -1].item() == eos_idx:
                candidates.append((seq, lang, score))
                continue
                
            sz = seq.size(1)
            tgt_mask = get_causal_mask(sz, device=device)
            
            # 4. Decode autoregressively without re-running the encoder
            char_logits, _ = underlying_model.decode(
                memory=memory,
                tgt_tokens=seq,
                tgt_langs=lang,
                src_pad_mask=src_pad_mask,
                tgt_mask=tgt_mask,
                tgt_pad_mask=None
            )
            
            log_probs = torch.log_softmax(char_logits[0, -1, :], dim=-1)
            topk_vals, topk_idxs = torch.topk(log_probs, beam_width)
            
            for val, idx in zip(topk_vals, topk_idxs):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
                new_lang = torch.cat([lang, torch.full((1, 1), target_lang_id, dtype=torch.long, device=device)], dim=-1)
                candidates.append((new_seq, new_lang, score + val.item()))
                
        candidates.sort(key=lambda x: x[2] / (max(1, x[0].size(1)) ** 0.75), reverse=True)
        beams = candidates[:beam_width]

        if all(seq[0, -1].item() == eos_idx for seq, _, _ in beams): break
        
    results = []
    for seq, _, score in beams:
        decoded = vocab.decode(seq[0].tolist())
        cleaned = [c for c in decoded if c not in vocab.special_tokens]
        results.append((cleaned, score))
    return results

def evaluate_reconstructions(model: nn.Module, dataloader: DataLoader, vocab, feature_matrix: torch.Tensor, target_lang_id: int, device: torch.device):
    model.eval()
    all_reconstructed, all_ground_truths = [], []
    
    # Corrected: Unpack src_pos as the third variable
    for src_tok, src_lng, src_pos, tgt_tok, _ in dataloader:
        src_tok = src_tok.to(device)
        src_lng = src_lng.to(device)
        src_pos = src_pos.to(device)
        
        predicted_ids = greedy_decode(model, src_tok, src_lng, src_pos, vocab, target_lang_id)
        
        pred_lists = predicted_ids.tolist()
        gold_lists = tgt_tok.tolist()
        
        for i in range(src_tok.size(0)):
            pred_chars_clean = [c for c in vocab.decode(pred_lists[i]) if c not in vocab.special_tokens]
            gold_chars_clean = [c for c in vocab.decode(gold_lists[i]) if c not in vocab.special_tokens]
            all_reconstructed.append(pred_chars_clean)
            all_ground_truths.append(gold_chars_clean)
            
    total_samples = len(all_reconstructed)
    exact_matches, total_ld, total_fld = 0, 0.0, 0.0
    fuzzy_1_matches, fuzzy_2_matches, stem_matches = 0, 0, 0
    cost_matrix = precompute_feature_cost_matrix(feature_matrix)

    for pred, gold in zip(all_reconstructed, all_ground_truths):
        if pred == gold: exact_matches += 1
        
        ld = edit_distance(pred, gold)
        total_ld += ld 
        total_fld += feature_weighted_edit_distance(pred, gold, vocab, cost_matrix)
        
        if ld <= 1: fuzzy_1_matches += 1
        if ld <= 2: fuzzy_2_matches += 1
        if pred[:3] == gold[:3]: stem_matches += 1

    return (exact_matches / total_samples, total_ld / total_samples, total_fld / total_samples,
            fuzzy_1_matches / total_samples, fuzzy_2_matches / total_samples, 
            all_reconstructed[:3], all_ground_truths[:3], stem_matches / total_samples)

# =====================================================================
# Training & Validation Execution (FeVeT-IA)
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Deep FeVeT Indo-Aryan Pipeline on hardware: {device}")

    # 1. OPTIMIZATION: Enable TensorFloat-32 (TF32) on Ampere (RTX 3070)
    # This automatically speeds up standard FP32 operations on Tensor Cores with zero loss in accuracy.
    torch.set_float32_matmul_precision('high')

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

    # 1. Download raw JAMBU datasets and parse into comparative etymon rows
    forms_raw, langs_raw, params_raw = download_jambu_data()
    segmenter = LinguisticSegmenter()
    df = process_jambu_to_dataframe_unified(forms_raw, langs_raw, params_raw, daughter_columns)
    
    # 2. Group split based on the CDIAL root (Parameter_ID)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=67)
    train_idx, val_idx = next(gss.split(df, groups=df['Parameter_ID']))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    print("\nBuilding Reconstruction Vocabulary...")
    train_raw_tokens = []
    for col in [target_column] + daughter_columns:
        for value in train_df[col].dropna():
            if isinstance(value, list):
                # Safely unpack list values
                for v in value:
                    if str(v) != "-":
                        train_raw_tokens.append(segmenter.tokenize(str(v)))
            else:
                if str(value) != "-":
                    train_raw_tokens.append(segmenter.tokenize(str(value)))

    vocab = ReconstructionVocab()
    vocab.build_vocab(train_raw_tokens, [])
    print(f"Total Unique Vocabulary Tokens: {len(vocab)}")

    train_dataset = JambuCognateDataset(train_df, target_column, daughter_columns, vocab, segmenter)
    val_dataset = JambuCognateDataset(val_df, target_column, daughter_columns, vocab, segmenter)

    pad_idx = vocab.token_to_idx["<PAD>"]

    train_collate = partial(collate_fn, pad_idx=pad_idx, lang_dropout_prob=0.20)
    val_collate = partial(collate_fn, pad_idx=pad_idx, lang_dropout_prob=0.0)

    train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=train_collate,
    pin_memory=True,
    num_workers=6,
    persistent_workers=True
    )

    val_loader = DataLoader(
    val_dataset, 
    batch_size=512, 
    shuffle=False, 
    collate_fn=val_collate,
    pin_memory=True,
    num_workers=6,
    persistent_workers=True
    )

    # Generate 30-D dual-compatible phonological matrix
    feature_matrix = build_phonological_matrix(vocab).to(device)
    print(f"Generated extended 30-D phonological matrix: {feature_matrix.shape}")

    model = FeVeTTransformer(
        vocab_size=len(vocab),
        num_features=FEATURE_DIM,
        feature_matrix=feature_matrix,
        num_languages=len(daughter_columns),
        d_model=128,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=2,
        dim_feedforward=256,
    ).to(device)

    if int(torch.__version__.split('.')[0]) >= 2:
        model = torch.compile(model, dynamic=True)

    criterion = FeVeTMultiTaskLoss(feature_matrix, alpha=0.5, pad_idx=pad_idx)
    
    epochs = 200
    max_learning_rate = 5e-4  # Peak learning rate for OneCycleLR

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_learning_rate / 10,
        weight_decay=0.05,
        betas=(0.9, 0.98)
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,             # Peak learning rate
        steps_per_epoch=len(train_loader),    # Number of batches per epoch
        epochs=epochs,                        # Total epochs planned
        pct_start=0.25,                       # 25% of time spent warming up
        div_factor=10.0,                      # Starts at 1e-4
        final_div_factor=100.0,               # Ends at 1e-6
        anneal_strategy='cos'                 # Cosine annealing
    )

    target_lang_id = train_dataset.lang_to_idx["TARGET"]

    start_alpha = 0.4
    end_alpha = 0.8
    decay_span = 150

    history = {
        "train_loss": [],
        "val_ld": []
    }

    # 1. Initialize trackers
    best_val_fld = float('inf')
    best_val_ld = float('inf')
    best_checkpoint_path = "best_fevet_ia_model.pt"
    latest_checkpoint_path = "latest_checkpoint.pt"
    patience = 50
    epochs_without_improvement = 0
    start_epoch = 1

    # 2. Check for an existing checkpoint to resume from
    if os.path.exists(latest_checkpoint_path):
        print(f"\n[INFO] Found checkpoint '{latest_checkpoint_path}'. Resuming training...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        
        # Restore states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore trackers
        start_epoch = checkpoint['epoch'] + 1
        best_val_fld = checkpoint['best_val_fld']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        
        print(f"Successfully restored states. Resuming from Epoch {start_epoch}.\n")

    # Determine native BF16 compatibility
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision_type = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using mixed precision dynamic type: {precision_type}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        if epoch <= decay_span:
            current_alpha = start_alpha - (start_alpha - end_alpha) * (epoch - 1) / (decay_span - 1)
        else:
            current_alpha = end_alpha

        criterion.alpha = current_alpha
        epoch_loss, epoch_ce_loss, epoch_bce_loss = 0.0, 0.0, 0.0

        # Corrected: Unpack src_pos as the third variable
        for src_tok, src_lng, src_pos, tgt_tok, tgt_lng in train_loader:
            src_tok = src_tok.to(device, non_blocking=True)
            src_lng = src_lng.to(device, non_blocking=True)
            src_pos = src_pos.to(device, non_blocking=True)
            tgt_tok = tgt_tok.to(device, non_blocking=True)
            tgt_lng = tgt_lng.to(device, non_blocking=True)

            tgt_input, tgt_output = tgt_tok[:, :-1], tgt_tok[:, 1:]
            tgt_lng_input = tgt_lng[:, :-1]

            src_pad_mask = make_bool_pad_mask(src_tok, pad_idx)
            tgt_pad_mask = make_bool_pad_mask(tgt_input, pad_idx)
            
            tgt_mask = get_causal_mask(tgt_input.size(1), device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=precision_type, enabled=(device.type == "cuda")):
                # Corrected: Pass src_pos explicitly as a keyword argument
                char_logits, feature_preds = model(
                    src_tokens=src_tok, 
                    src_langs=src_lng, 
                    src_pos=src_pos, 
                    tgt_tokens=tgt_input, 
                    tgt_langs=tgt_lng_input, 
                    src_pad_mask=src_pad_mask, 
                    tgt_mask=tgt_mask, 
                    tgt_pad_mask=tgt_pad_mask
                )
                loss, loss_ce, loss_bce = criterion(char_logits, feature_preds, tgt_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_bce_loss += loss_bce.item()

        mean_loss = epoch_loss / len(train_loader)
        mean_ce = epoch_ce_loss / len(train_loader)
        mean_bce = epoch_bce_loss / len(train_loader)

        val_acc, val_ld, val_fld, val_f1, val_f2, pred_sample, gold_sample, stem_3 = evaluate_reconstructions(
            model, val_loader, vocab, feature_matrix, target_lang_id, device
        )

        history["train_loss"].append(mean_loss)
        history["val_ld"].append(val_ld)

        is_best = ""
        if val_fld < best_val_fld:
            best_val_fld = val_fld
            torch.save(model.state_dict(), best_checkpoint_path)
            is_best = " [BEST MODEL SAVED]"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        full_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_fld': best_val_fld,
            'epochs_without_improvement': epochs_without_improvement
        }
        torch.save(full_checkpoint, latest_checkpoint_path)

        if epoch == 1 or epoch % 10 == 0 or "BEST" in is_best:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | Alpha: {criterion.alpha:.3f}{is_best}")
            print(f"  [Train] Loss: {mean_loss:.4f} (CE: {mean_ce:.4f}, BCE: {mean_bce:.4f})")
            print(f"  [Val]   Exact: {val_acc * 100:.2f}% | Fuzzy-1: {val_f1 * 100:.2f}% | Fuzzy-2: {val_f2 * 100:.2f}% | LD: {val_ld:.4f} | FLD: {val_fld:.4f}")
            print(f"  [Sample Pred] : {' '.join(pred_sample[0])}")
            print(f"  [Sample Gold] : {' '.join(gold_sample[0])}")
            print(f"  [Stem-3 Match]: {stem_3 * 100:.2f}%\n")

        if epochs_without_improvement >= patience:
            print(f"\n[Early Stopping Triggered] No improvement in Validation Feature Edit Distance for {patience} epochs.")
            print(f"Terminating training early at Epoch {epoch}.")
            break

    # Restore the best model checkpoint for final verification
    model.load_state_dict(torch.load(best_checkpoint_path))
    model.eval()