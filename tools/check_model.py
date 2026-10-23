"""checks the new token model before starting a real training run"""

from pathlib import Path
import sys

# keep the project folder as the base after moving this helper
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time
import json
import torch
from subword_model import SubwordConfig, SubwordLanguageModel


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU check.")
    torch.set_num_threads(4)
    torch.manual_seed(448)
    model = SubwordLanguageModel(SubwordConfig(**json.loads((ROOT/'configs/token_14m.json').read_text()))).cuda()
    x = torch.randint(6, 4096, (8, 256), device="cuda")
    y = torch.randint(6, 4096, (8, 256), device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    started = time.perf_counter()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        assert logits.shape == (8, 256, 4096) and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    model.eval()
    a = x[:2, :64].clone()
    b = a.clone()
    b[:, 32:] = (b[:, 32:] + 1) % 4096
    with torch.no_grad():
        first, _ = model(a)
        second, _ = model(b)
    assert torch.allclose(first[:, :32], second[:, :32], atol=1e-5, rtol=1e-5)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Passed: output shape, finite loss/gradients, optimizer updates and causal mask.")
    print("Twenty throwaway update steps took", round(seconds, 2), "seconds including setup overhead.")
    print("This checked model is discarded; the real training run will start fresh.")


if __name__ == "__main__":
    main()
