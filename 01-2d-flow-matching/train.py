"""Train a 2D velocity field and save weights, config, and a training plot."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import DATASETS, sample_target
from network import VelocityField
from visualize import save_train_plot


def parse_train_args():
    parser = argparse.ArgumentParser(description="Train independent-coupling linear 2D flow matching")
    parser.add_argument("--dataset", choices=DATASETS, default="moons")
    parser.add_argument("--steps", type=int, default=35000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def train(model, args):
    """Fit the velocity field with conditional flow matching."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    model.train()
    for step in range(args.steps):
        x0 = torch.randn(args.batch_size, 2)
        x1 = sample_target(args.batch_size, args.dataset)
        t = torch.rand(args.batch_size, 1)
        xt = (1 - t) * x0 + t * x1  # [batch_size, 2]; t=0 is noise, t=1 is data.
        target_velocity = x1 - x0  # [batch_size, 2]
        loss = F.mse_loss(model(xt, t), target_velocity)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; try a lower learning rate")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step == 0 or (step + 1) % 500 == 0 or step + 1 == args.steps:
            print(f"step {step+1:5d}/{args.steps}  MSE {sum(losses[-100:])/len(losses[-100:]):.4f}", flush=True)
    return losses


def save_results(out, model, config, losses, target):
    """Save weights, configuration, target points, and training loss."""
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": config}, out / "model.pt")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_train_plot(out, target, losses)
    print(f"Saved to {out.resolve()}")


def main():
    args = parse_train_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    out = Path("runs") / f"{args.dataset}-{time.time_ns()}"
    out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "out": str(out)}

    model = VelocityField(args.hidden)
    losses = train(model, args)
    target = sample_target(2000, args.dataset)  # Target data for plotting; no model sampling.
    save_results(out, model, config, losses, target)


if __name__ == "__main__":
    main()
