# Nigerian Proverbs LLM 🇳🇬

Fine-tuning a LLaMA-family model with **LoRA** (via Hugging Face `transformers` + `peft`) on a curated dataset of Nigerian proverbs (Yoruba, Igbo, and Hausa), so the model can generate culturally grounded proverbs and explain their meanings.

## Why this project?

Large language models are trained overwhelmingly on Western/English-centric text, and often produce shallow or inaccurate output when asked about African oral traditions. This project explores lightweight fine-tuning (LoRA) as a way to inject culturally specific knowledge into an open LLM without the cost of full fine-tuning.

## What's in here

```
nigerian-proverbs-llm/
├── data/
│   └── proverbs.jsonl          # 52 curated proverbs: instruction/input/output triples
├── scripts/
│   ├── augment.py              # Paraphrase-augments the dataset (52 → 103 examples)
│   ├── preprocess.py           # Formats data into a training prompt template
│   ├── train_lora.py           # LoRA fine-tuning with 4-bit quantization (QLoRA-style)
│   └── inference.py            # Load base model + adapter, generate text (CLI)
├── app.py                      # Gradio web demo for interactive querying
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

`data/proverbs.jsonl` contains **52 curated proverbs** across Yoruba, Igbo, and Hausa, each with:
- the original proverb,
- an English translation,
- a plain-language explanation of its meaning/usage.

`scripts/augment.py` expands this to **103 training examples** by generating paraphrased instruction variants for each proverb (e.g. "Share a Yoruba proverb about patience" → "What's a Yoruba proverb about patience?"). This is a standard instruction-tuning technique — it does **not** invent new proverbs, only new ways of asking for the same one, which helps the model generalize to varied phrasing at inference time.

Contributions of additional proverbs (with accurate translations and attribution to the language/region) are welcome via pull request — the seed set is intentionally a starting point, not a finished corpus. Native-speaker review is encouraged before treating any entry as authoritative; a few (noted in `data/proverbs.jsonl`) are drawn from proverbs referenced in Chinua Achebe's writing.

## Setup

```bash
git clone https://github.com/<your-username>/nigerian-proverbs-llm.git
cd nigerian-proverbs-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You'll need access to a GPU (a free Colab T4 is enough for a small model like `meta-llama/Llama-3.2-1B` with 4-bit quantization). For gated models on Hugging Face, run `huggingface-cli login` and accept the model license on the model's page first.

## Usage

**1. Augment the dataset (52 → 103 examples)**
```bash
python scripts/augment.py --input data/proverbs.jsonl --output data/proverbs_augmented.jsonl
```

**2. Preprocess into a training prompt format**
```bash
python scripts/preprocess.py --input data/proverbs_augmented.jsonl --output data/proverbs_formatted.jsonl
```

**3. Fine-tune with LoRA**
```bash
python scripts/train_lora.py \
  --base_model meta-llama/Llama-3.2-1B \
  --data_path data/proverbs_formatted.jsonl \
  --output_dir outputs/proverbs-lora \
  --epochs 3
```

**4. Generate (command line)**
```bash
python scripts/inference.py \
  --base_model meta-llama/Llama-3.2-1B \
  --adapter_dir outputs/proverbs-lora \
  --prompt "Share an Igbo proverb about perseverance and explain its meaning."
```

**5. Interactive demo (Gradio)**
```bash
python app.py --base_model meta-llama/Llama-3.2-1B --adapter_dir outputs/proverbs-lora
```
This launches a local web UI at `http://127.0.0.1:7860` where you can type requests and get generated proverbs + explanations. Pass `--share` to get a temporary public link (useful for a demo GIF/video in your README). If no trained adapter is found, it falls back to the base model so you can test the UI before training finishes.

## Notes & limitations

- 52 unique proverbs is still a modest base for robust fine-tuning; the augmentation step helps the model generalize instruction phrasing, but doesn't add new cultural knowledge. Growing the *unique* proverb count remains the highest-leverage next step.
- Translations/meanings should be reviewed by native speakers before being treated as authoritative; proverbs often carry regional variation and nuance that a short gloss can't fully capture. A few entries reflect proverbs referenced in Chinua Achebe's writing rather than direct fieldwork.
- This is a research/learning project, not a production system.

## Roadmap

- [ ] Expand unique proverb count to 200+ across more Nigerian languages (e.g. Efik, Ijaw, Edo)
- [ ] Add an evaluation script comparing base vs. fine-tuned outputs
- [ ] Publish the LoRA adapter weights on the Hugging Face Hub
- [x] Simple Gradio demo for interactive querying
- [x] Data augmentation script for instruction paraphrasing

## License

MIT — see [LICENSE](LICENSE).
