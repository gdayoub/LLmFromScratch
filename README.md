# LLmFromScratch

I built this small learning project to understand how language models work. I trained a 13.9 million parameter model from scratch with PyTorch, then continued training it on basic conversations and animal questions. I used AI coding assistance while building and evaluating the project.

My model generates one token at a time. It does not look up answers in the dataset or call a hosted model API. It remains experimental and can produce incorrect or nonsensical answers.

## How I organized the project

- `model.py`: attention and feed-forward blocks plus the original byte prototype
- `subword_model.py`: the active transformer
- `train.py`: next-token training, validation, and checkpoint saving
- `sample_subword.py`: terminal chat
- `tools/prepare_data.py`: build the teaching data and token cache
- `tools/check_model.py`: check shapes, gradients, updates, and causal masking
- `tools/evaluate.py`: evaluate generated answers
- `data/conversations/`: readable training and evaluation examples
- `data/sources/animal_facts.json`: 28 animal records and source notes

I used 6 transformer blocks, 6 attention heads, an embedding width of 384, a vocabulary of 4096 tokens, and a context window of 256 tokens. During animal training, I mixed earlier basic skills with animal questions, comparisons, follow-ups, and explicitly fictional description-reading exercises.

## Setup

I developed this project with Python 3.12 and an NVIDIA CUDA GPU. To run it, create an environment and install PyTorch appropriate for your CUDA setup, along with the tokenizer package:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch tokenizers
.\.venv\Scripts\python.exe -B tools/check_model.py
.\.venv\Scripts\python.exe -B tools/prepare_data.py
```

I recorded my original development environment in `requirements.txt`, including the CUDA-specific PyTorch build. Your setup may need a different PyTorch installation.

I kept the tokenizer frozen as I continued training. My preparation script refuses to overwrite an existing token cache. I left my checkpoints and original English corpus out of GitHub; see `data/english/README.txt` for the English data notes.

To try my training pipeline with a fresh randomly initialized model and only the included teaching curriculum:

```powershell
.\.venv\Scripts\python.exe -B train.py --model-config configs/token_14m.json --tokenized-data artifacts/animals --data data/conversations --dialogue-batches --english-fraction 0 --out runs/animals --steps 4000 --batch-size 20 --lr 0.0001 --eval-every 100 --patience 8
```

I first trained my original model on TinyStories, then basic conversations, then animals. The command above demonstrates the training pipeline, but reproducing my reported results requires those earlier stages and their data.

## Chat and evaluation

Once I have a checkpoint, I use these commands to chat with the model and evaluate its answers:

```powershell
.\.venv\Scripts\python.exe -B sample_subword.py --checkpoint runs/animals/best.pt
.\.venv\Scripts\python.exe -B tools/evaluate.py --checkpoint runs/animals/best.pt --split validation --report-prefix animals
```

I use `/reset` to clear conversation history and `/quit` to exit. I restart the chat whenever I want to load updated weights. To stop training and save progress, I press Ctrl+C in its terminal or create a `STOP` file inside the active run directory.

## What I measured

I measured an improvement from 2/33 to 28/33 strict reference matches on the same reserved animal exercises. My animal checkpoint matched 12/12 short fictional-description exercises with held-out names and food/home pairings. On my earlier basic diagnostic, the total stayed at 21/38, although some individual skills improved and others regressed.

I treat these results as a limited learning experiment. The exercises are small, related, and synthetic, and strict reference matching can reject valid paraphrases. My model still produced nonsense on some reworded animal questions. I have now inspected the final exercises, so I would need a fresh unseen benchmark for future tuning.

I used [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) for English training. I included CDC and Smithsonian references for nuanced animal claims in the animal fact records. I do not claim ownership of third-party source materials.
