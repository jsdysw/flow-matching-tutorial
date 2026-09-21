"""Display RGB images with fixed pixel scaling."""

import matplotlib
import torch
from torchvision.utils import make_grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def show_images(ax, samples, title, nrow=8, column_labels=None):
    images = ((samples.detach().cpu() + 1) / 2).clamp(0, 1)  # Clamp for display only.
    grid = make_grid(images, nrow=nrow, padding=2, pad_value=0.5)
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_title(title)
    ax.axis("off")
    if column_labels is not None:
        ax.axis("on")
        ax.set_xticks([18 + 34 * i for i in range(len(column_labels))], column_labels, fontsize=9)
        ax.set_yticks([])
        ax.set_xlabel("Requested class")
        ax.tick_params(bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)


def save_train_plot(out, images, losses):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    show_images(axes[0], images, "True CIFAR-10")
    axes[1].plot(range(1, len(losses) + 1), losses, linewidth=0.6)
    axes[1].set(title="Training loss", xlabel="Training step", ylabel="MSE")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=150)
    plt.close(fig)


def save_eval_plot(out, path, classes, samples_per_class):
    steps = len(path) - 1
    samples = path[-1].reshape(len(classes), samples_per_class, 3, 32, 32)
    samples = samples.transpose(0, 1).flatten(0, 1)  # Rows: noise samples; columns: requested classes.
    width = max(4, len(classes) * 1.05)
    fig, ax = plt.subplots(figsize=(width, max(3, samples_per_class * 1.05)))
    show_images(ax, samples, f"Conditional CIFAR-10 · Euler {steps} steps",
                nrow=len(classes), column_labels=classes)
    ax.set_ylabel("Same initial noise within each row")
    fig.tight_layout()
    fig.savefig(out / f"euler_{steps}.png", dpi=150)
    plt.close(fig)

    indices = sorted({round(t * steps) for t in (0, 0.25, 0.5, 0.75, 1)})
    snapshots = torch.stack([path[index, ::samples_per_class] for index in indices]).flatten(0, 1)
    fig, ax = plt.subplots(figsize=(width, 5))
    show_images(ax, snapshots, "Noise → image", nrow=len(classes), column_labels=classes)
    ax.set_yticks([18 + 34 * i for i in range(len(indices))],
                  [f"{index / steps:.2f}" for index in indices])
    ax.set_ylabel("Time t")
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(out / f"trajectory_{steps}.png", dpi=150)
    plt.close(fig)
