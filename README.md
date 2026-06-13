# FeVeT-IA: Feature Vector Transformer for Indo-Aryan Phonological Reconstruction

FeVeT-IA is a computational historical linguistics framework designed for **automated phonological reconstruction** and **cognate reflex prediction** across Indo-Aryan (IA) languages. Moving away from traditional sequence models that treat speech sounds as discrete, isolated tokens, this project models phonology by breaking down segments into continuous, structured phonetic feature vectors.

The core architecture adapts the **Feature Vector Transformer (FeVeT)** framework and augments it with a deep language model prior to enforce valid historical phonotactics during sequence generation.

---

## Core Features & Architecture Quirks

### 1. Continuous Phonetic Feature Vectors

Instead of mapping an International Phonetic Alphabet (IPA) character to an arbitrary index token, `fevet_ia.py` maps segments into a high-dimensional articulatory feature space.

* Sounds are decomposed into binary/multi-valued vectors representing phonological primitives (e.g., `[+high]`, `[-back]`, `[+voice]`, `[+nasal]`).
* This allows the neural network to understand *why* sounds shift over time based on actual articulatory distance rather than treating characters as orthogonal entities.

### 2. Hybrid Autoregressive LSTM Prior

A unique quirk of this setup is the injection of `sanskrit_lstm_prior.pt`. Historical sound changes often produce candidate words that look plausible to an unconstrained Transformer but break strict historical constraints. The LSTM prior tracks ancestral phonotactics (such as permissible consonant clusters or vowel gradations in Sanskrit), acting as a regularizer during the decoding phase.

### 3. Custom Multitask Objective

The network simultaneously optimizes two loss functions during training:

1. **Feature Prediction Loss:** A cross-entropy/MSE loss over the individual phonetic feature dimensions.
2. **Character Classification Loss:** A standard soft-max cross-entropy loss over the final decoded IPA character space.

---

### Script Execution Pipelines

* **Model Definition (`fevet_ia.py`):** Implements the multi-headed attention mechanism over continuous feature spaces.
* **Beam Search Decoding (`inference_beam_search.py`):** Performs sequence generation. Instead of a greedy argmax approach, it explores multiple high-probability trajectories through a custom search tree weighted by the prior.
* **Visualizing Embeddings (`visualize_space.py`):** Extracts weights from the phonetic embedding layer and generates a 2D map of the learned sound relationships.

---

## Inference & Decoding Mechanics

The system's decoding engine goes beyond generic sequence generation. Given a source cognate set vector, the sequence probability $P(Y|X)$ during beam search decoding is conditioned jointly on the Transformer's encoder state and the phonotactical language prior:

$$P(Y|X) = \prod_{t=1}^{T} P(y_t \mid y_{\lt t}, X)^{\alpha} \cdot P_{\text{prior}}(y_t \mid y_{\lt t})^{\beta}$$

Where $\alpha$ and $\beta$ are hyperparameters scaling the influence of the core translation model versus the Sanskrit phonotactic prior.

---

## Citations & References

* **Indo-Aryan Cognate Dataset**
> Arora, A., Farris, A., Basu, S., & Kolichala, S. (2023). *Jambu: A historical linguistic database for South Asian languages.* In Proceedings of the 20th SIGMORPHON workshop on Computational Research in Phonetics, Phonology, and Morphology (pp. 68–77). Association for Computational Linguistics. <https://doi.org/10.18653/v1/2023.sigmorphon-1.8>


* **Feature Vector Transformer (FeVeT) Architecture:**
> Wientzek, T. (2025). *Using feature vectors for automated phonological reconstruction and reflex prediction.* Open Research Europe, 5(174). [https://doi.org/10.12386/ore.2025.5-174](https://open-research-europe.ec.europa.eu/articles/5-174)


* **Cognate Reflex and Sequence Alignment Frameworks:**
> List, J.-M., Forkel, R., & Hill, N. (2022). *A New Framework for Fast Automated Phonological Reconstruction Using Trimmed Alignments and Sound Correspondence Patterns.* Proceedings of the 3rd Workshop on Computational Approaches to Historical Language Change, 89–96. [https://doi.org/10.18653/v1/2022.lchange-1.9](https://doi.org/10.18653/v1/2022.lchange-1.9)


* **Cross-Linguistic Transcription Data Standards:**
> List, J.-M., Anderson, C., Tresoldi, T., Rzymski, C. & Forkel, R. (2024). *CLTS: Cross-Linguistic Transcription Systems (v2.3.0).* Zenodo. [https://doi.org/10.5281/zenodo.10997741](https://doi.org/10.5281/zenodo.10997741)
