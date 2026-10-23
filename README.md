# LLmFromScratch

I started this project because I wanted to understand what actually happens when a language model learns. I also wanted something I could open in front of my TA, talk to, and explain without hiding behind a pile of scripts I did not understand.

I built and trained it on my Lenovo 7i with an RTX 5070 laptop GPU. The current model has about 13.9 million parameters. It started from random weights, learned English, and then continued training on basic conversations and animal questions.

I used AI coding assistance throughout this project. That helped me get things running, but it also became part of the challenge: sometimes the code moved forward faster than my understanding. I kept asking for explanations before changes because having a working file was not the same as knowing why it worked. This repository is a record of that learning process, including the parts that did not work.

## What went wrong along the way

My first idea was to make the model talk like Radahn from Elden Ring. I wanted it to mention his horse and the scarlet rot so it would feel like the character. The problem was that the model was still struggling to speak English. Character references could make a reply sound convincing without making it answer the question. I eventually dropped that direction and focused on basic conversation.

The early replies were often garbled, irrelevant, or repetitive. I thought adding a lot more English data and training longer would solve that. Switching from bytes to word-piece tokens and moving to a larger model helped with parts of the problem, but neither automatically made it a good conversational model. A readable sentence could still be a completely wrong answer.

One frustrating lesson was that a falling training loss did not mean the model was getting better at new questions. In several runs, it kept improving on its practice examples while validation got worse. I learned why saving the best checkpoint matters more than simply keeping the last one.

I also let the project get too messy. There were too many experiments, datasets, and checking scripts, and I could not easily explain which ones mattered. I simplified the active tools to three jobs: prepare the data, check the model, and evaluate its answers. That made the project easier for me to follow and explain.

Even testing had a practical lesson: an open terminal chat keeps the weights it loaded. At one point I was questioning whether VS Code had the new model when an old chat was still explicitly loading an earlier checkpoint. Updating a file does not update a model already running in memory.

## What started to work

Short, direct teaching examples worked better for this small model than the broader conversation data I had tried. But I did not want a demo that only repeated one answer for one exact question. I started varying the wording, including follow-up conversations, and keeping separate questions for evaluation.

Animals gave the project a manageable focus. The curriculum covers 28 animals: their sounds, groups, habitats, features, and comparisons. I also included clearly fictional creatures with different foods and homes supplied in the question. Holding out some names and combinations gave me a small way to check whether the model could use the description instead of only recalling a familiar fact.

Seeing it answer "What animal barks?" correctly was encouraging. Seeing it read a description of an unfamiliar fictional creature and answer what it eats was more interesting to me. It was still a narrow exercise, but it gave me something more meaningful to test than a rehearsed greeting.

## What I learned

- Parameters are learned numbers. Adding training data changes their values; it does not automatically add more parameters.
- English text practice and learning to answer a question are different training tasks.
- Better examples can help more than repeating the same examples for longer.
- A model can imitate the shape of an answer without getting its meaning right.
- New training can weaken earlier skills, so I mixed in older lessons and English and checked them again.
- Once I inspect a test and use its failures to improve the model, I need a fresh test for an independent final check.

I am still learning these ideas. I would not describe this as a small version of ChatGPT that understands everything. My model generates one token at a time from its learned weights and context, without looking up dataset answers or calling a hosted model API. It can handle some basic questions, but unfamiliar wording can still produce nonsense. Being able to show both the successes and failures is part of what makes this useful to me as a learning project.

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

I call the terminal chatbot **Clanker**. On Windows, I double-click `Start Clanker.cmd` in the project folder. It uses the project's Python environment and loads the animal checkpoint automatically. From the VS Code PowerShell terminal, I can also run:

```powershell
& ".\Start Clanker.cmd"
```

The launcher requires the local environment and trained checkpoint described above. The name changes the terminal display; it does not retrain the model.

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
