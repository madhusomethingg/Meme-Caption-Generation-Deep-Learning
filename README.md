# 🐸 Automated Meme Generation Using Deep Learning
### Image-Conditioned Caption Generation with LSTMs and Transformers

> An end-to-end deep learning pipeline that generates witty meme captions from input images — benchmarking 8 models across LSTM and Transformer architectures at both word and character level, using the DeepHumor framework on 900K+ meme captions.

---

## 📌 Overview

Memes are one of the most uniquely human forms of communication — they combine visual context with language in ways that require understanding both. Automating that process is a genuinely hard AI problem: the model needs to look at an image, understand its cultural context, and generate a caption that is not just grammatically correct but actually funny.

This project builds an image-conditioned caption generation system trained on 900K real memes across 200 templates. It compares 8 model variants — LSTM and Transformer decoders, with and without template labels, at both word and character level — and generates memes using beam search decoding with temperature sampling.

| Goal | Approach |
|---|---|
| Generate contextually relevant meme captions from images | Image-conditioned sequence generation with CNN encoder |
| Compare word-level vs character-level caption modeling | 8 models across both tokenization strategies |
| Evaluate LSTM vs Transformer decoder architectures | Direct benchmarking on same dataset and splits |
| Support both template-based and open-ended generation | Label-conditioned and label-free model variants |

---

## 📂 Dataset

**Memes900K**  
Source: [Google Drive](https://drive.google.com/drive/folders/1VGIEnsMHctHMopW3fHaL5TZ7tf2OjPzd) *(too large for GitHub)*

- **Size:** 900,000+ meme captions across 300 templates (200 used for training)
- **Splits:** Train / Validation / Test
- **Vocabulary:** Word-level (WordPunct tokenizer) and Character-level tokenizers, both with min_df=5
- **Images:** Resized to 224×224, normalized with ImageNet mean/std

---

## 🔧 Tech Stack

| Category | Libraries / Tools |
|----------|-----------|
| Deep Learning | `PyTorch` |
| Image Encoding | `torchvision` (ResNet-based CNN backbone) |
| Caption Decoding | `CaptioningLSTM`, `CaptioningTransformer` (DeepHumor) |
| Text Processing | `WordPunctTokenizer`, `CharTokenizer` |
| Generation | Beam search + temperature sampling + top-k filtering |
| Codebase | [DeepHumor](https://github.com/ilya16/deephumor) |

---

## 🗂️ Repository Structure

```
├── memegeneration.ipynb     # Full pipeline — data, models, generation
├── Dataset/
│   └── datasetcrawled       # Google Drive link to full dataset
└── README.md
```

---

## 🔬 Methodology

### 1. Data Preparation
- Dataset loaded from Google Drive (`memes900k.zip`)
- Word and character vocabularies built from `captions_train.txt` with `min_df=5`
- `MemeDataset` built for each split (train/val/test) at both word and character level
- Images preloaded and transformed: resize → tensor → ImageNet normalize

### 2. Model Architecture

The pipeline follows an encoder-decoder structure:
- **Encoder:** CNN (ResNet-based) extracts visual features from the input image
- **Decoder:** Generates the caption token-by-token conditioned on image features

Eight model variants are compared:

| # | Model | Tokenization | Label-Conditioned |
|---|-------|:------------:|:-----------------:|
| 1 | LSTM Decoder | Word | ❌ |
| 2 | LSTM Decoder with Labels | Word | ✅ |
| 3 | Transformer (global image embedding) | Word | ❌ |
| 4 | Transformer (spatial image features) | Word | ❌ |
| 5 | LSTM Decoder | Character | ❌ |
| 6 | LSTM Decoder with Labels | Character | ✅ |
| 7 | Transformer (global image embedding) | Character | ❌ |
| 8 | Transformer (spatial image features) | Character | ❌ |

> **Key Design Choice:** Spatial Transformer models receive patch-level image features rather than a single global embedding — allowing the model to attend to specific regions of the image during caption generation, rather than treating the entire image as a single context vector.

### 3. Caption Generation

Captions are generated using **beam search** with temperature and top-k sampling:

```python
get_a_meme(
    model=w_transformer_model,
    img_torch=img_torch,
    img_pil=img_pil,
    caption=caption,       # optional seed caption
    T=1.0,                 # temperature
    beam_size=10,
    top_k=70,
    mode='word'
)
```

- Both template images (from the dataset) and new custom images are supported
- Optional `caption` seed allows partial caption completion
- Label conditioning allows generating captions in the style of a specific meme template (e.g., "Willy Wonka", "Matrix Morpheus")

---

## 📊 Key Results

- All 8 model variants successfully generate contextually relevant meme captions
- Transformer models with spatial features produce more visually grounded captions compared to global-embedding variants
- Character-level models capture fine-grained stylistic patterns (punctuation, capitalization) that word-level models miss
- Label-conditioned models produce captions more consistent with the cultural tone of specific meme templates
- Beam search with temperature > 1.0 increases caption creativity at the cost of coherence — tunable per use case

---

## ⚠️ Limitations

- Trained on 200 of 300 available templates — captions for unseen templates may be generic
- Humor is subjective — automated evaluation metrics (BLEU, perplexity) don't fully capture whether a meme is actually funny
- Model requires GPU for inference — character-level Transformer is especially compute-intensive
- Dataset sourced from internet memes, which may reflect cultural biases present in that content

---

## 🔮 Future Work

- **Multimodal LLMs** — fine-tune a vision-language model (e.g., LLaVA, BLIP-2) on meme data for richer cultural understanding
- **Human evaluation** — build a rating interface to collect human funniness scores for model comparison
- **Diffusion-based meme generation** — combine caption generation with image synthesis to create entirely new meme formats
- **Retrieval-augmented generation** — retrieve culturally relevant meme templates before generating captions

---

## 🚀 Getting Started

```bash
pip install torch torchvision pillow
```

1. Open `memegeneration.ipynb` in Google Colab (GPU recommended)
2. The notebook auto-downloads the codebase from [DeepHumor](https://github.com/ilya16/deephumor) and datasets from Google Drive
3. Run sections in order: Data → Models → Generation
4. To generate from a custom image, place it in `images_inference/` and update the `img_pil` path

---

## 👤 Author

Madhumitha Rajagopal

---

## 📄 License

This project is for educational and research purposes.
