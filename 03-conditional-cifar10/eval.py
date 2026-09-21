"""Generate selected CIFAR-10 classes with Euler integration."""

import argparse
from pathlib import Path

import torch

from dataset import CLASSES
from network import VelocityField
from visualize import save_eval_plot


def parse_eval_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=list(CLASSES))
    parser.add_argument("--ode-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


@torch.no_grad()
def integrate(model, x0, labels, steps=128):
    x = x0.clone()  # [samples, 3, 32, 32]
    path = [x.cpu().clone()]
    dt = 1 / steps
    for i in range(steps):
        t = x.new_full((len(x), 1), i * dt)  # [samples, 1]
        x = x + dt * model(x, t, labels)
        path.append(x.cpu().clone())
    return torch.stack(path)  # [steps + 1, samples, 3, 32, 32]


def main():
    args = parse_eval_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VelocityField(checkpoint["config"]["channels"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rng = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(args.samples_per_class, 3, 32, 32, generator=rng)
    x0 = noise.repeat(len(args.classes), 1, 1, 1).to(args.device)
    labels = torch.tensor([CLASSES.index(name) for name in args.classes])
    labels = labels.repeat_interleave(args.samples_per_class).to(args.device)
    path = integrate(model, x0, labels, args.ode_steps)
    out = Path(args.checkpoint).parent
    save_eval_plot(out, path, args.classes, args.samples_per_class)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
