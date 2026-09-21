# 03 · Conditional CIFAR-10 Flow Matching

Generate 32×32 RGB images of ten object classes with a class-conditional U-Net.
This is an initial baseline for GPU training; sample quality still needs a full training run.

## Setup

```bash
cd 03-conditional-cifar10
conda create -n flow-matching python=3.11 -y
conda activate flow-matching
pip install -r requirements.txt
```

If the environment already exists, activate it and install the requirements.
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) downloads to `data/` on the first training run.
Images stay uint8 in CPU memory; each training batch is converted to [B, 3, 32, 32] floats in [-1, 1].

## Model

The MNIST model's conditioning and flow-matching loss stay the same.
The U-Net adds one downsampling level and uses 64 base channels:

| Block | Output shape |
| --- | --- |
| Input | [B, 3, 32, 32] |
| down1 | [B, 64, 32, 32] |
| down2 | [B, 128, 16, 16] |
| down3 | [B, 256, 8, 8] |
| middle | [B, 256, 4, 4] |
| up3 | [B, 256, 8, 8] |
| up2 | [B, 128, 16, 16] |
| up1 | [B, 64, 32, 32] |
| Velocity | [B, 3, 32, 32] |

Each block receives `time_embedding(t) + class_embedding(y)` with shape [B, 128].
Encoder-to-decoder skips concatenate channels; each block also has a residual addition.

## Train

```bash
python train.py --device cuda --steps 50000 --batch-size 128
```

Use `cuda` on the GPU server, `mps` on Mac, or `cpu` (default).
Learning rate 3e-4 · gradient clipping 1.0. Reduce `--batch-size` if GPU memory is insufficient.
50,000 steps is a starting training budget, not a guaranteed quality threshold.

Saves `model.pt`, `config.json`, and `summary.png` (real images and loss) to `runs/<run>/`.
Intermediate `step_<n>.pt` checkpoints are saved every 5,000 steps for evaluation.
Checkpoints contain weights and configuration, not optimizer state; training starts from scratch.

## Eval

```bash
# Ten columns, five samples per class
python eval.py runs/<run>/model.pt --device cuda --ode-steps 128

# Select classes by name
python eval.py runs/<run>/model.pt --device cuda --classes cat dog ship
```

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
Each row shares the same initial noise across classes.
Saves `euler_128.png` and `trajectory_128.png` beside the checkpoint.
Trajectory columns show one shared noise sample becoming each class, from t=0 to t=1.
Rerunning eval with the same step count replaces these PNGs. Data and runs are ignored by Git.
