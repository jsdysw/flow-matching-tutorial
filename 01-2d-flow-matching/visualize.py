"""Save training and generation results as PNGs."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_train_plot(out, target, losses):
    target = target.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(*target.T, s=3, alpha=0.45)
    axes[0].set(title="True data", xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
    axes[1].plot(range(1, len(losses) + 1), losses, linewidth=0.6)
    axes[1].set(title="Training loss", xlabel="Training step", ylabel="MSE")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=150)
    plt.close(fig)


def save_eval_plot(out, path, target, steps, limit):
    """Show noise-to-data snapshots and trajectories with persistent particle colors."""
    path, target = path.cpu().numpy(), target.cpu().numpy()
    # Each particle keeps its initial x-coordinate color throughout the flow.
    colors = plt.cm.viridis(plt.Normalize(-limit, limit)(path[0, :, 0]))
    indices = sorted({round(t * steps) for t in (0, 0.25, 0.5, 0.75, 1)})
    columns = len(indices) + 2
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4.8))
    snapshots = list(axes[:-2])
    target_ax, trajectory_ax = axes[-2:]
    for ax, index in zip(snapshots, indices):
        t = index / steps
        points = path[index]
        ax.scatter(*points.T, c=colors, s=4, alpha=0.55)
        label = "Noise" if t == 0 else "Generated" if t == 1 else "Flow"
        ax.set_title(f"{label}: t = {t:.2f}")
    target_ax.scatter(*target.T, c="coral", s=4, alpha=0.55)
    target_ax.set_title("True data")

    # Alternate 6 outer starts and 6 inner starts across 12 angular sectors.
    # Inner starts are the points closest to radius 1 from the origin.
    # Selection uses only starting positions, not how well particles reach the target.
    trajectory_ax.scatter(*target.T, c="lightgray", s=3, alpha=0.35)
    radius = np.linalg.norm(path[0], axis=1)
    angle = np.arctan2(path[0, :, 1], path[0, :, 0]) % (2 * np.pi)
    sectors = (angle / (2 * np.pi) * 12).astype(int)
    for sector in range(12):
        candidates = np.flatnonzero(sectors == sector)
        if len(candidates) == 0:
            continue
        if sector % 2 == 0:
            i = candidates[radius[candidates].argmax()]
        else:
            i = candidates[np.abs(radius[candidates] - 1).argmin()]
        points = path[:, i]
        color = colors[i]
        trajectory_ax.plot(*points.T, color=color, linewidth=1.2, alpha=0.8)
        trajectory_ax.scatter(*points[0], facecolors="none", edgecolors=[color], s=35)
        trajectory_ax.scatter(*points[-1], c=[color], s=25)
        # Arrows at available snapshot times indicate the direction of time.
        for end in indices[1:]:
            trajectory_ax.annotate("", xy=points[end], xytext=points[end - 1],
                                   arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    trajectory_ax.set_title("Trajectories: 6 outer + 6 inner starts")
    trajectory_ax.set_xlabel("Open circle: noise · Filled circle: generated")

    for ax in snapshots + [target_ax, trajectory_ax]:
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.grid(alpha=0.15)
    fig.suptitle(f"Euler | {steps} steps | {steps} NFE | Gaussian noise → data\n"
                 "Colors identify particles by initial x; same axes across all panels")
    fig.tight_layout()
    fig.savefig(out / f"euler_{steps}.png", dpi=150)
    plt.close(fig)
