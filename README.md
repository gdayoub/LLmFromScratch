# LLmFromScratch

A small learning project by George Dayoub: a 13.9 million parameter language model trained from scratch with PyTorch, then continued on basic conversations and animal questions. Developed with AI coding assistance.

The model generates one token at a time. It does not look up answers in the dataset or call a hosted model API. It remains experimental and can produce incorrect or nonsensical answers.

## What's here

- `model.py`: attention and feed-forward blocks plus the original byte prototype
- `subword_model.py`: the active transformer
- `train.py`: next-token training, validation, and checkpoint saving
- `sample_subword.py`: terminal chat
- `tools/prepare_data.py`: build the teaching data and token cache
- `tools/check_model.py`: check shapes, gradients, updates, and causal masking
- `tools/evaluate.py`: evaluate generated answers
- `data/conversations/`: readable training and evaluation examples
- `data/sources/animal_facts.json`: 28 animal records and source notes

Architecture: 6 blocks, 6 attention heads, width 384, vocabulary 4096, context 256 tokens. Animal training mixes earlier basic skills with animal questions, comparisons, follow-ups, and explicitly fictional description-reading exercises.

## Setup

Python 3.12 and an NVIDIA CUDA GPU were used. Create an environment and install PyTorch appropriate for your CUDA setup, then the tokenizer package:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch tokenizers
.\.venv\Scripts\python.exe -B tools/check_model.py
.\.venv\Scripts\python.exe -B tools/prepare_data.py
```

`requirements.txt` records the original development environment, including its CUDA-specific PyTorch build; it is not a universal installation command.

The included tokenizer is frozen. Preparation refuses to overwrite an existing token cache. Checkpoints and the original English corpus are excluded from GitHub. See `data/english/README.txt` for English data notes.

To train a fresh randomly initialized model using only the included teaching curriculum:

```powershell
.\.venv\Scripts\python.exe -B train.py --model-config configs/token_14m.json --tokenized-data artifacts/animals --data data/conversations --dialogue-batches --english-fraction 0 --out runs/animals --steps 4000 --batch-size 20 --lr 0.0001 --eval-every 100 --patience 8
```

That command demonstrates the pipeline; it does not reproduce the English-pretrained model's reported performance. The original model first learned from TinyStories, then basic conversations, then animals. Reproducing that result requires those earlier stages and their data.

## Chat and evaluation

After creating a checkpoint:

```powershell
.\.venv\Scripts\python.exe -B sample_subword.py --checkpoint runs/animals/best.pt
.\.venv\Scripts\python.exe -B tools/evaluate.py --checkpoint runs/animals/best.pt --split validation --report-prefix animals
```

`/reset` clears conversation history; `/quit` exits. Restart chat to load changed weights. Training can be stopped with Ctrl+C in its terminal, which saves progress, or by creating a `STOP` file inside the active run directory.

## Recorded experiment

The original animal checkpoint improved from 2/33 to 28/33 strict reference matches on the same reserved animal exercises. This included 12/12 short fictional-description exercises with held-out names and food/home pairings. The earlier basic diagnostic stayed at 21/38 overall, with individual regressions and gains.

These are small, related synthetic exercises. Reference matching can reject valid paraphrases, and the results do not establish broad understanding. Some reworded animal questions still produced nonsense. The final exercises have now been inspected and should not be reused as an unseen benchmark for future tuning.

English source: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Nuanced animal claims have CDC and Smithsonian references in the animal fact records. No license is asserted over third-party source materials.
