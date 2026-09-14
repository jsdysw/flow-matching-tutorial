"""Load a checkpoint and save generated samples as a PNG."""

import argparse
from pathlib import Path

import torch

from dataset import sample_target
from network import VelocityField
from visualize import save_eval_plot


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Generate samples from a saved model")
    parser.add_argument("checkpoint")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--ode-steps", type=int, default=128)
    return parser.parse_args()


@torch.no_grad()
def integrate(model, x0, steps=128):
    """Euler ODE integration on [0, 1]; return [steps+1, batch, 2]."""
    x = x0.clone()
    path = [x.clone()]
    dt = 1.0 / steps
    for i in range(steps):
        t = x.new_full((len(x), 1), i * dt)
        v = model(x, t)
        x = x + dt * v
        path.append(x.clone())
    return torch.stack(path)


def generate(model, config, args):
    """Generate samples with the requested Euler steps and target points for plotting."""
    model.eval()
    rng = torch.Generator().manual_seed(config["seed"] + 10000)
    x0 = torch.randn(args.samples, 2, generator=rng)
    target = sample_target(args.samples, config["dataset"], generator=rng)
    path = integrate(model, x0, args.ode_steps)
    return path, target


def main():
    args = parse_eval_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]

    model = VelocityField(config["hidden"])
    model.load_state_dict(checkpoint["model"])
    path, target = generate(model, config, args)

    out = Path(args.checkpoint).parent
    limit = max(4, int(target.abs().max().ceil()) + 1, int(path.abs().max().ceil()) + 1)
    save_eval_plot(out, path, target, len(path) - 1, limit)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
