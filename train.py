"""trains my small models and keeps checkpoints so i can pick them up later"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from model import ModelConfig, RadahnLanguageModel
from subword_model import SubwordConfig, SubwordLanguageModel

# find the project folder even if i run this from somewhere else
ROOT = Path(__file__).resolve().parent


def dialogue_tensors(path, context_length, device):
    # keep each visitor question with its answer instead of starting halfway through
    inputs, targets = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ids, learn = [], []
        for index, message in enumerate(record["messages"]):
            expected = "user" if index % 2 == 0 else "assistant"
            if message["role"] != expected or not message["content"].strip():
                raise ValueError("Dialogue needs alternating nonempty visitor and assistant turns.")
            prefix = ("Visitor:" if expected == "user" else "Radahn:").encode("utf-8")
            body = (" " + message["content"] + "\n").encode("utf-8")
            ids.extend(prefix)
            learn.extend([False] * len(prefix))
            ids.extend(body)
            learn.extend([expected == "assistant"] * len(body))
        ending = b"<|end_conversation|>\n"
        ids.extend(ending)
        learn.extend([True] * len(ending))
        if len(ids) - 1 > context_length:
            raise ValueError("Conversation exceeds context; split it explicitly before dialogue training.")
        # fill unused positions and ignore visitor text when measuring this loss
        x = torch.zeros(context_length, dtype=torch.long)
        y = torch.full((context_length,), -100, dtype=torch.long)
        sequence = torch.tensor(ids, dtype=torch.long)
        mask = torch.tensor(learn, dtype=torch.bool)
        length = len(ids) - 1
        x[:length] = sequence[:-1]
        y[:length] = torch.where(mask[1:], sequence[1:], -100)
        inputs.append(x)
        targets.append(y)
    if not inputs:
        raise ValueError("No conversations found.")
    return torch.stack(inputs).to(device), torch.stack(targets).to(device)


def prediction_loss(model, x, y, per_example=False, reply_prefix_weight=1.0):
    if not per_example:
        return model(x, y)[1]
    logits, _ = model(x)
    # average each example over its real targets so padding doesnt change its weight
    losses = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="none")
    valid = y != -100
    weights = valid.float()
    if reply_prefix_weight != 1.0:
        # choosing the opening words should matter more than finishing a familiar answer
        positions = torch.arange(y.shape[1], device=y.device).unsqueeze(0).expand_as(y)
        previous = F.pad(valid[:, :-1], (1, 0), value=False)
        starts = valid & ~previous
        last_start = torch.where(starts, positions, -1).cummax(dim=1).values
        # ordinary english chunks have no masked visitor tokens and keep normal weights
        dialogue_rows = (~valid).any(dim=1, keepdim=True)
        opening = valid & dialogue_rows & (positions - last_start < 8)
        weights = weights * torch.where(opening, reply_prefix_weight, 1.0)
    counts = weights.sum(dim=1).clamp_min(1)
    return ((losses * weights).sum(dim=1) / counts).mean()


def arguments():
    # these options let me change a run without editing the code every time
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "radahn_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "radahn_v2_001")
    # steps means total weight updates not full trips through the dataset
    parser.add_argument("--steps", type=int, default=1000)
    # each batch has this many chunks of text
    parser.add_argument("--batch-size", type=int, default=8)
    # learning rate controls how big the weight changes can be
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    # seed helps repeat the random choices but different hardware can still differ
    parser.add_argument("--seed", type=int, default=448)
    parser.add_argument("--resume", action="store_true")
    # start a new experiment from my own learned weights without changing the old run
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--english-data", type=Path)
    # this is the share of chunks from english not a share of files
    parser.add_argument("--english-fraction", type=float, default=0.75)
    parser.add_argument("--log-every", type=int, default=100)
    # whole conversations with loss only on replies and the ending marker
    parser.add_argument("--dialogue-batches", action="store_true")
    # this selects the new tokenizer cache and larger token model instead of bytes
    parser.add_argument("--tokenized-data", type=Path)
    parser.add_argument("--reply-prefix-weight", type=float, default=1.0)
    # keep character practice separate so it cant crowd out ordinary conversations
    parser.add_argument("--persona-data", type=Path)
    parser.add_argument("--persona-fraction", type=float, default=0.0)
    # a json file lets me try more capacity without changing the old defaults
    parser.add_argument("--model-config", type=Path)
    # stop when separate examples stop improving instead of memorizing forever
    parser.add_argument("--patience", type=int, default=0)
    args = parser.parse_args()
    if args.patience < 0:
        parser.error("Patience cannot be negative.")
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batches) < 1 or args.lr <= 0:
        parser.error("Steps, batch size, evaluation settings and learning rate must be positive.")
    if not 0 <= args.english_fraction <= 1 or args.log_every < 1:
        parser.error("English fraction must be 0..1 and log frequency must be positive.")
    if args.resume and args.init_from:
        parser.error("Choose either resume or init-from, not both.")
    if args.init_from and args.init_from.resolve().parent == args.out.resolve():
        parser.error("Use a new output folder when continuing from another run.")
    if args.reply_prefix_weight < 1 or (args.reply_prefix_weight != 1 and not args.dialogue_batches):
        parser.error("Reply prefix weight needs dialogue batches and must be at least one.")
    if not 0 <= args.persona_fraction <= 1:
        parser.error("Persona fraction must be 0..1.")
    if bool(args.persona_data) != bool(args.persona_fraction):
        parser.error("Persona data and a positive persona fraction must be used together.")
    if args.persona_data and (not args.tokenized_data or not args.dialogue_batches):
        parser.error("Persona mixing needs tokenized dialogue batches.")
    if args.persona_fraction + (args.english_fraction if args.english_data else 0) > 1:
        parser.error("English and persona fractions cannot exceed one together.")
    if args.model_config and not args.tokenized_data:
        parser.error("Model config files are only supported for the token model.")
    return args


def main():
    args = arguments()
    # dont silently fall back to the cpu and make training way slower
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently train on the CPU.")
    # limit cpu helper threads and set the starting random states
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # allow faster matrix math on supported hardware
    torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    # latest is for resuming and best is picked using validation loss
    latest_path = args.out / "latest.pt"
    best_path = args.out / "best.pt"
    stop_path = args.out / "STOP"
    # these checks keep me from overwriting a run or ignoring a stop request
    if stop_path.exists():
        raise RuntimeError("STOP file exists; remove it before starting or resuming.")
    if latest_path.exists() and not args.resume:
        raise RuntimeError("This run already has a checkpoint; use --resume or a new --out folder.")
    if args.resume and not latest_path.exists():
        raise RuntimeError("No latest.pt exists to resume.")

    # this makes a fresh random model before any optional checkpoint gets loaded
    token_metadata = None
    tokenizer_path = None
    if args.tokenized_data:
        token_metadata = json.loads((args.tokenized_data / "manifest.json").read_text(encoding="utf-8"))
        tokenizer_path = (args.tokenized_data / "tokenizer.json").resolve()
        config_values = json.loads(args.model_config.read_text(encoding="utf-8")) if args.model_config else {}
        if (args.init_from or args.resume) and not args.model_config:
            # continuing a run must keep the saved architecture
            source_path = latest_path if args.resume else args.init_from
            source_config = torch.load(source_path, map_location="cpu", weights_only=True)
            config_values = source_config["config"]
            del source_config
        config_values.setdefault("vocab_size", token_metadata["vocab_size"])
        config = SubwordConfig(**config_values)
        if config.vocab_size != token_metadata["vocab_size"] or config.context_length != token_metadata.get("context_length", 256):
            raise ValueError("Model vocabulary and context must match the prepared cache.")
        model = SubwordLanguageModel(config).cuda()
    else:
        config = ModelConfig()
        model = RadahnLanguageModel(config).cuda()
    # adamw uses gradients to update weights and gently penalizes large weights
    # the betas control its moving averages of gradients and squared gradients
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    # keep batch sampling separate from random choices inside the model
    training_rng = torch.Generator().manual_seed(args.seed + 1)
    data = {}
    data_hashes = {}
    # keep sources and splits separate so a bigger file cant decide our mixing ratio
    token_cache = None
    if args.tokenized_data:
        cache_path = args.tokenized_data / "tokens.pt"
        data_hashes = {"tokenizer": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                       "token_cache": hashlib.sha256(cache_path.read_bytes()).hexdigest()}
        if data_hashes["tokenizer"] != token_metadata["tokenizer_sha256"] or data_hashes["token_cache"] != token_metadata["cache_sha256"]:
            raise ValueError("Tokenizer or token cache changed after preparation.")
        token_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        data = {key: value.to("cuda") for key, value in token_cache["streams"].items()}
    else:
        sources = [("", args.data)]
        if args.english_data:
            sources.append(("english_", args.english_data))
        for prefix, folder in sources:
            for split in ("train", "validation", "test"):
                key = prefix + split
                raw = (folder / f"{split}.txt").read_bytes()
                if len(raw) <= config.context_length:
                    raise ValueError(f"{key} needs more than {config.context_length} bytes.")
                # integer byte ids live on the gpu and dont contain pretrained knowledge
                data[key] = torch.tensor(list(raw), dtype=torch.long, device="cuda")
                data_hashes[key] = hashlib.sha256(raw).hexdigest()
    english_count = round(args.batch_size * args.english_fraction) if args.english_data else 0
    persona_count = round(args.batch_size * args.persona_fraction)
    if english_count + persona_count > args.batch_size or (args.persona_data and not persona_count):
        raise ValueError("Rounded mixture must fit the batch and include persona examples when requested.")
    sampler = {"english_chunks": english_count, "radahn_chunks": args.batch_size - english_count - persona_count,
               "batch_size": args.batch_size, "english_fraction": english_count / args.batch_size,
               "radahn_objective": "assistant_only" if args.dialogue_batches else "all_text"}
    sampler["reply_prefix_weight"] = args.reply_prefix_weight
    if args.persona_data:
        sampler["persona_chunks"] = persona_count
        sampler["persona_fraction"] = persona_count / args.batch_size
    dialogues = {}
    if args.dialogue_batches:
        if token_cache is not None:
            dialogues = {key: (x.to("cuda"), y.to("cuda")) for key, (x, y) in token_cache["dialogues"].items()}
        else:
            for split in ("train", "validation", "test"):
                path = args.data / f"{split}.jsonl"
                dialogues[split] = dialogue_tensors(path, config.context_length, "cuda")
                data_hashes["dialogue_" + split] = hashlib.sha256(path.read_bytes()).hexdigest()

    persona_dialogues = {}
    if args.persona_data:
        # token ids must mean the same words in every source
        persona_metadata = json.loads((args.persona_data / "manifest.json").read_text(encoding="utf-8"))
        persona_cache_path = args.persona_data / "tokens.pt"
        persona_hash = hashlib.sha256(persona_cache_path.read_bytes()).hexdigest()
        persona_tokenizer_hash = hashlib.sha256((args.persona_data / "tokenizer.json").read_bytes()).hexdigest()
        if persona_tokenizer_hash != data_hashes["tokenizer"] or persona_hash != persona_metadata["cache_sha256"]:
            raise ValueError("Persona cache or tokenizer does not match.")
        persona_cache = torch.load(persona_cache_path, map_location="cpu", weights_only=True)
        for split, (x, y) in persona_cache["dialogues"].items():
            if x.shape != y.shape or x.shape[1] != config.context_length:
                raise ValueError("Persona context does not match the model.")
            persona_dialogues[split] = (x.to("cuda"), y.to("cuda"))
        data_hashes["persona_cache"] = persona_hash

    use_bfloat16 = torch.cuda.is_bf16_supported()

    def precision():
        # use smaller numbers for some calculations to save memory and work
        # the actual model parameters still stay float32
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()

    def batch(split, rng, source_name=None):
        # labels are the same chunk shifted forward by one token
        counts = [("", sampler["radahn_chunks"]), ("english_", sampler["english_chunks"]), ("persona_", persona_count)]
        if source_name is not None:
            counts = [({"english": "english_", "persona": "persona_"}.get(source_name, ""), args.batch_size)]
        inputs, targets = [], []
        for prefix, count in counts:
            if not count:
                continue
            if prefix == "persona_":
                x, y = persona_dialogues[split]
                chosen = torch.randint(len(x), (count,), generator=rng).tolist()
                inputs.extend(x[chosen].unbind(0))
                targets.extend(y[chosen].unbind(0))
                continue
            if not prefix and args.dialogue_batches:
                # padding comes after valid text so causal attention cant leak it backwards
                x, y = dialogues[split]
                chosen = torch.randint(len(x), (count,), generator=rng).tolist()
                inputs.extend(x[chosen].unbind(0))
                targets.extend(y[chosen].unbind(0))
                continue
            source = data[prefix + split]
            # pick random starting spots within this source rather than concatenating files
            starts = torch.randint(len(source) - config.context_length, (count,), generator=rng).tolist()
            inputs.extend(source[i:i + config.context_length] for i in starts)
            # each spot gets the next token as its answer
            targets.extend(source[i + 1:i + config.context_length + 1] for i in starts)
        return torch.stack(inputs), torch.stack(targets)

    # dont build gradient tracking while just checking predictions
    @torch.no_grad()
    def evaluate(split, source_name=None):
        was_training = model.training
        # eval turns dropout off so checks arent adding training noise
        model.eval()
        # fixed evaluation chunks make comparisons less noisy between checks
        source_offset = 100 if source_name == "english" else 0
        rng = torch.Generator().manual_seed(args.seed + {"train": 10, "validation": 20, "test": 30}[split] + source_offset)
        values = []
        for _ in range(args.eval_batches):
            x, y = batch(split, rng, source_name)
            with precision():
                loss = prediction_loss(model, x, y, args.dialogue_batches, args.reply_prefix_weight)
            # item turns a single tensor value into a normal python number
            values.append(loss.item())
        # put the model back in whatever mode it had before this check
        model.train(was_training)
        return sum(values) / len(values)

    started = time.perf_counter()
    start_step = 0
    step = 0
    # infinity makes the first real validation result count as an improvement
    best_validation = math.inf
    initial_metrics = None
    last_metrics = None
    lineage = []
    if args.init_from:
        # transfer only my own weights and reset adamw for the changed data mixture
        checkpoint = torch.load(args.init_from, map_location="cuda", weights_only=True)
        if checkpoint["config"] != asdict(config):
            raise ValueError("Source checkpoint architecture does not match this model.")
        # matching shapes alone cannot tell us whether token ids mean the same thing
        if args.tokenized_data and checkpoint.get("data_hashes", {}).get("tokenizer") != data_hashes["tokenizer"]:
            raise ValueError("Source checkpoint uses a different tokenizer.")
        model.load_state_dict(checkpoint["model_state"])
        lineage = checkpoint.get("lineage", []) + [{"path": str(args.init_from.resolve()),
                   "sha256": hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
                   "step": checkpoint["step"]}]
    if args.resume:
        # load only checkpoints i trust and put their tensors on the gpu
        checkpoint = torch.load(latest_path, map_location="cuda", weights_only=True)
        if checkpoint["config"] != asdict(config) or checkpoint["data_hashes"] != data_hashes:
            raise ValueError("Configuration or data changed since this checkpoint.")
        saved_sampler = dict(checkpoint.get("sampler", sampler))
        saved_sampler.setdefault("radahn_objective", "all_text")
        saved_sampler.setdefault("reply_prefix_weight", 1.0)
        if saved_sampler != sampler:
            raise ValueError("Keep the same source mixture when resuming; start a new stage to change it.")
        lineage = checkpoint.get("lineage", [])
        model.load_state_dict(checkpoint["model_state"])
        # weights alone arent enough to resume adamw where it left off
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = start_step = checkpoint["step"]
        best_validation = checkpoint["best_validation"]
        initial_metrics = checkpoint["initial_metrics"]
        last_metrics = checkpoint["last_metrics"]
        # restore random states too so sampling continues from its saved state
        training_rng.set_state(checkpoint["training_rng"].cpu())
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])

    # append logs on resume and flush each line so progress is visible right away
    log_file = (args.out / "training.log").open("a", encoding="utf-8", buffering=1)
    metrics_file = (args.out / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")

    def save(path):
        # write a temporary file first so a partial write doesnt replace the old save
        temporary = path.with_suffix(".tmp")
        # keep weights plus training state and settings so the save is useful later
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "config": asdict(config), "step": step, "best_validation": best_validation,
            "initial_metrics": initial_metrics, "last_metrics": last_metrics,
            "data_hashes": data_hashes, "training_rng": training_rng.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "sampler": sampler, "lineage": lineage,
            "model_kind": "subword" if args.tokenized_data else "byte",
            "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
            "tokenized_data": str(args.tokenized_data.resolve()) if args.tokenized_data else None,
        }, temporary)
        temporary.replace(path)

    def measure():
        # nonlocal means update these values from the surrounding main function
        nonlocal best_validation, last_metrics
        # train loss shows fitting while validation checks separate text
        train_loss = evaluate("train")
        radahn_validation = evaluate("validation", "radahn")
        english_validation = evaluate("validation", "english") if args.english_data else None
        fraction = sampler["english_fraction"]
        persona_fraction = sampler.get("persona_fraction", 0)
        persona_validation = evaluate("validation", "persona") if args.persona_data else None
        validation_loss = (1 - fraction - persona_fraction) * radahn_validation
        if english_validation is not None:
            validation_loss += fraction * english_validation
        if persona_validation is not None:
            validation_loss += persona_fraction * persona_validation
        last_metrics = {"step": step, "train_loss": train_loss, "validation_loss": validation_loss,
                        "radahn_validation_loss": radahn_validation,
                        "english_validation_loss": english_validation,
                        "seconds_this_session": time.perf_counter() - started}
        if persona_validation is not None:
            last_metrics["persona_validation_loss"] = persona_validation
        metrics_file.write(json.dumps(last_metrics) + "\n")
        log(f"step {step:4d} | train {train_loss:.4f} | validation {validation_loss:.4f}")
        if english_validation is not None:
            log(f"  Dialogue validation {radahn_validation:.4f} | English validation {english_validation:.4f}")
        if persona_validation is not None:
            log(f"  Persona validation {persona_validation:.4f} | general dialogue is the legacy Radahn metric above")
        # choose best using validation not the test set or the training score
        improved = validation_loss < best_validation
        if improved:
            best_validation = validation_loss
        return improved

    # record the setup so i can explain and compare this experiment later
    run_config = {"model": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
                  "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "data_hashes": data_hashes, "data_lengths": {k: len(v) for k, v in data.items()},
                  "data_unit": "tokens" if args.tokenized_data else "bytes",
                  "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                  "precision": "bfloat16" if use_bfloat16 else "float32",
                  "objective": ("next-token" if args.tokenized_data else "next-byte") + (" cross entropy with assistant-only dialogue targets and all English targets" if args.dialogue_batches else " cross entropy on all training text positions"),
                  "external_pretrained_weights": False, "sampler": sampler, "lineage": lineage,
                  "starting_weights": "own checkpoint" if args.init_from or args.resume else "random"}
    (args.out / ("resume_config.json" if args.resume else "run_config.json")).write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    log(f"GPU: {run_config['gpu']} | parameters: {run_config['parameters']:,} | context: {config.context_length} {run_config['data_unit']}")
    log(f"Data: {run_config['data_lengths']} | precision: {run_config['precision']}")
    starting_label = "Resuming" if args.resume else "Continuing my own weights" if args.init_from else "Starting from random weights"
    log(f"{starting_label} at stage step {step}; target {args.steps} steps")
    log(f"Batch mixture: {sampler}")
    completed = False
    stale_checks = 0
    early_stopped = False
    try:
        if not args.resume:
            # measure the starting weights first so this stage has an honest before score
            improved = measure()
            initial_metrics = dict(last_metrics)
            if improved:
                save(best_path)
            save(latest_path)
        # train mode enables dropout but does not update any weights by itself
        model.train()
        for next_step in range(start_step + 1, args.steps + 1):
            if stop_path.exists():
                log("STOP requested; saving progress.")
                break
            # warm up the learning rate rather than taking full size steps at once
            warmup = min(1.0, next_step / 50)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup
            x, y = batch("train", training_rng)
            # clear old gradients or they would add up across batches
            optimizer.zero_grad(set_to_none=True)
            with precision():
                loss = prediction_loss(model, x, y, args.dialogue_batches, args.reply_prefix_weight)
            # stop if the math turns into invalid or infinite values
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; stopping this run.")
            # backward works out how each weight contributed to the error
            loss.backward()
            # limit the overall gradient size to help avoid huge updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            # this is the line that actually changes the learned weights
            optimizer.step()
            step = next_step
            if step % args.log_every == 0:
                elapsed = time.perf_counter() - started
                log(f"progress {step}/{args.steps} | batch loss {loss.item():.4f} | {elapsed:.1f}s elapsed")
            # check separate text regularly and keep both best and latest saves
            if step % args.eval_every == 0 or step == args.steps:
                improved = measure()
                if improved:
                    save(best_path)
                save(latest_path)
                stale_checks = 0 if improved else stale_checks + 1
                if args.patience and stale_checks >= args.patience:
                    log("Validation stopped improving so training is stopping early.")
                    early_stopped = True
                    break
        completed = step >= args.steps
    except KeyboardInterrupt:
        # ctrl c requests a save instead of just throwing away this session
        log("Interrupted; saving progress.")
    finally:
        # finally runs on normal finish or an interrupted training loop
        save(latest_path)
        summary = {"status": "completed" if completed else "early_stopped" if early_stopped else "stopped", "step": step,
                   "total_lineage_steps": sum(parent["step"] for parent in lineage) + step,
                   "lineage": lineage, "sampler": sampler,
                   "best_validation_loss": best_validation, "initial_metrics": initial_metrics,
                   "latest_metrics": last_metrics, "seconds_this_session": time.perf_counter() - started}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"Checkpoint saved: {latest_path}")
        log_file.close()
        metrics_file.close()


# run main only when i launch this file directly
if __name__ == "__main__":
    main()
