"""talks to the new model with the same token rules it learned during training"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from subword_model import SubwordConfig, SubwordLanguageModel

ROOT = Path(__file__).resolve().parent


def load_model(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "subword":
        raise ValueError("This program needs a subword checkpoint rather than a byte model.")
    tokenizer_path = Path(checkpoint["tokenizer_path"])
    # the data cleanup moved the tokenizer but its hash must still match
    if not tokenizer_path.exists():
        tokenizer_path = ROOT / "configs/tokenizer.json"
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != checkpoint["data_hashes"]["tokenizer"]:
        raise ValueError("Tokenizer does not match these weights.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = SubwordLanguageModel(SubwordConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, checkpoint


@torch.inference_mode()
def reply(model, tokenizer, question, history=None, max_tokens=90, temperature=0.7, top_k=30):
    ids = {name: tokenizer.token_to_id(name) for name in ["<|pad|>", "<|unk|>", "<|eos|>", "<|visitor|>", "<|radahn|>", "<|story|>"]}
    history = list(history) if history else []
    # the old radahn marker is just the assistant token id now
    # keeping its id lets me reuse my english weights
    # roles use the same explicit markers as the training examples
    prompt = history + [ids["<|visitor|>"]] + tokenizer.encode(question).ids + [ids["<|radahn|>"]]
    sequence, generated = list(prompt), []
    device = next(model.parameters()).device
    for _ in range(max_tokens):
        x = torch.tensor([sequence[-model.config.context_length:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        scores = logits[0, -1].float() / temperature
        # role and padding tokens are structural controls not words to show in a reply
        for name, value in ids.items():
            if name != "<|eos|>":
                scores[value] = -torch.inf
        values, indices = torch.topk(scores, min(top_k, len(scores)))
        selected = torch.multinomial(torch.softmax(values, dim=-1), 1)
        token = indices[selected].item()
        if token == ids["<|eos|>"]:
            break
        generated.append(token)
        sequence.append(token)
    text = tokenizer.decode(generated).strip()
    return text, prompt + generated + [ids["<|eos|>"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "animals" / "best.pt")
    parser.add_argument("--prompt")
    # a little less randomness keeps this tiny model closer to its learned wording
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.temperature <= 0 or args.max_tokens < 1 or not 1 <= args.top_k <= 4096:
        parser.error("Use positive temperature and token limit; top-k must be 1..4096.")
    torch.set_num_threads(4)
    torch.manual_seed(448)
    model, tokenizer, checkpoint = load_model(args.checkpoint)
    total_steps = checkpoint['step'] + sum(parent['step'] for parent in checkpoint.get('lineage', []))
    print("Clanker")
    print(f"Loaded {args.checkpoint.parent.name} at {total_steps:,} total training steps; small experimental conversation model.")
    if args.probe:
        results = []
        for question in ["Hello!", "How are you doing?", "I like playing video games.", "I had a rough day."]:
            answer, _ = reply(model, tokenizer, question, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
            results.append({"prompt": question, "response": answer})
            print(json.dumps(results[-1], ensure_ascii=True), flush=True)
        report = {"stage_step": checkpoint["step"], "lineage": checkpoint.get("lineage", []),
                  "temperature": args.temperature, "top_k": args.top_k, "max_tokens": args.max_tokens, "seed": 448,
                  "samples": results, "retrieval_or_scripted_answers": False}
        (args.checkpoint.parent / "probe_samples.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    if args.prompt is not None:
        print(reply(model, tokenizer, args.prompt, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)[0])
        return
    history = []
    print("Type a message; /quit exits and /reset clears the conversation.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "/quit":
            break
        if question == "/reset":
            history = []
            continue
        if question:
            answer, history = reply(model, tokenizer, question, history, args.max_tokens, args.temperature, args.top_k)
            print("Clanker:", answer)


if __name__ == "__main__":
    main()
