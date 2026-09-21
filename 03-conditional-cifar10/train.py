"""Train a class-conditional CIFAR-10 flow."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import load_cifar10
from network import VelocityField
from visualize import save_train_plot


def parse_train_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def train(model, images, labels, args, out):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    model.train()
    for step in range(args.steps):
        indices = torch.randint(len(images), (args.batch_size,))  # [B]
        x1 = images[indices].to(args.device).float() / 127.5 - 1  # [B, 3, 32, 32], pixels in [-1, 1]
        y = labels[indices].to(args.device)  # [B], labels paired with x1
        x0 = torch.randn_like(x1)  # [B, 3, 32, 32], Gaussian noise
        t = torch.rand(args.batch_size, 1, device=args.device)  # [B, 1]
        time = t[:, :, None, None]  # [B, 1, 1, 1], broadcast over channels and pixels
        xt = (1 - time) * x0 + time * x1
        target_velocity = x1 - x0
        loss = F.mse_loss(model(xt, t, y), target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # E.g., if all gradients have L2 norm 10, divide each by 10.
        optimizer.step()
        losses.append(loss.item())
        if step == 0 or (step + 1) % 500 == 0 or step + 1 == args.steps:
            print(f"step {step+1}/{args.steps}  MSE {sum(losses[-100:])/len(losses[-100:]):.4f}", flush=True)
        if (step + 1) % 5000 == 0 and step + 1 < args.steps:
            save_checkpoint(out / f"step_{step+1}.pt", model, args, step + 1)
    return losses


def save_checkpoint(path, model, args, trained_steps):
    """Save model weights and settings for inference."""
    config = {**vars(args), "architecture": "conditional_unet", "grad_clip": 1.0,
              "trained_steps": trained_steps}
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": config}, path)
    return config


def main():
    args = parse_train_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    images, labels = load_cifar10()
    out = Path("runs") / f"cifar10-conditional-{time.time_ns()}"
    out.mkdir(parents=True)
    model = VelocityField(args.channels).to(args.device)
    losses = train(model, images, labels, args, out)
    config = save_checkpoint(out / "model.pt", model, args, args.steps)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_train_plot(out, images[:64].float() / 127.5 - 1, losses)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
