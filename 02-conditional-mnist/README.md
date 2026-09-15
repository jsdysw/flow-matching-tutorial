# 02 · Conditional MNIST Flow Matching

Generate digits 0–9 with a class-conditional U-Net.
Each block receives `time_embedding(t) + label_embedding(y)`.
Images stay `[B, 1, 28, 28]`, with pixels scaled to [-1, 1].

## Model

Default channel counts are shown below; image tensors use [B, C, H, W], where B is the batch size.
Dashed arrows supply the same time/class condition to all five blocks.

```mermaid
flowchart TB
    t["Time t · [B, 1]"] --> time["Linear 1→128 → SiLU → Linear 128→128<br/>[B, 128]"]
    y["Digit y · [B], values 0–9"] --> label["Embedding(10, 128)<br/>[B, 128]"]
    time --> condition["Add embeddings · [B, 128]"]
    label --> condition

    x["Input x_t<br/>[B, 1, 28, 28]"] --> down1["down1 · ConditionalBlock · 1→32 channels<br/>[B, 32, 28, 28]"]
    down1 -->|"AvgPool 2×2 · [B, 32, 14, 14]"| down2["down2 · ConditionalBlock · 32→64 channels<br/>[B, 64, 14, 14]"]
    down2 -->|"AvgPool 2×2 · [B, 64, 7, 7]"| middle["middle · ConditionalBlock · 64→128 channels<br/>[B, 128, 7, 7]"]
    middle -->|"Nearest upsample ×2 · [B, 128, 14, 14]"| cat2["Concatenate along channels<br/>[B, 192, 14, 14]"]
    down2 -->|"Skip · [B, 64, 14, 14]"| cat2
    cat2 --> up2["up2 · ConditionalBlock · 192→64 channels<br/>[B, 64, 14, 14]"]
    up2 -->|"Nearest upsample ×2 · [B, 64, 28, 28]"| cat1["Concatenate along channels<br/>[B, 96, 28, 28]"]
    down1 -->|"Skip · [B, 32, 28, 28]"| cat1
    cat1 --> up1["up1 · ConditionalBlock · 96→32 channels<br/>[B, 32, 28, 28]"]
    up1 --> output["Conv 1×1 · 32→1 channels<br/>Velocity v(x_t, t, y) · [B, 1, 28, 28]"]

    condition -.-> down1
    condition -.-> down2
    condition -.-> middle
    condition -.-> up2
    condition -.-> up1

    classDef block fill:#e8f1ff,stroke:#4875b5,color:#182b49;
    classDef cond fill:#fff3d6,stroke:#b88625,color:#493617;
    classDef skip fill:#e8f6ee,stroke:#498966,color:#203d2d;
    class down1,down2,middle,up2,up1 block;
    class t,y,time,label,condition cond;
    class cat1,cat2 skip;
```

The output is a velocity field; Euler integration turns noise into a digit.

### Inside a ConditionalBlock

```mermaid
flowchart TB
    x["Input x<br/>[B, C_in, H, W]"] --> conv1["Conv 3×3 · C_in → C_out<br/>[B, C_out, H, W]"]
    c["Time embedding + class embedding<br/>[B, 128]"] --> linear["Linear · 128 → C_out"]
    linear --> broadcast["Reshape to [B, C_out, 1, 1]<br/>Broadcast across H × W"]
    conv1 --> addCondition(("+"))
    broadcast --> addCondition
    addCondition --> norm1["GroupNorm · 8 groups<br/>SiLU"]
    norm1 --> conv2["Conv 3×3 · C_out → C_out<br/>GroupNorm · 8 groups"]
    x --> skip["Residual: Conv 1×1<br/>C_in → C_out"]
    conv2 --> addResidual(("+"))
    skip --> addResidual
    addResidual --> activation["SiLU"]
    activation --> out["Output<br/>[B, C_out, H, W]"]

    classDef feature fill:#e8f1ff,stroke:#4875b5,color:#182b49;
    classDef condition fill:#fff3d6,stroke:#b88625,color:#493617;
    classDef residual fill:#e8f6ee,stroke:#498966,color:#203d2d;
    class x,conv1,norm1,conv2,activation,out feature;
    class c,linear,broadcast,addCondition condition;
    class skip,addResidual residual;
```

The first `+` adds the projected condition at every pixel. The second `+` adds the residual features element by element.
Both are additions; the U-Net's encoder-to-decoder skips use concatenation. This block keeps H × W unchanged.

## Setup

```bash
cd 02-conditional-mnist
conda activate flow-matching
pip install -r requirements.txt
```

MNIST downloads to `data/` on the first training run.

## Train

```bash
python train.py --steps 3000 --device mps
```

Use `mps` on Mac, `cuda` on NVIDIA, or `cpu` (default).
Batch size 128 · base channels 32 · learning rate 3e-4 · gradient clipping 1.0.
Saves `model.pt`, `config.json`, and `summary.png` to `runs/`.
Longer runs save intermediate inference checkpoints every 5,000 steps.

## Eval

Use a checkpoint trained with this conditional model.

```bash
# All digits: 10 columns, 5 samples per digit
python eval.py runs/<run>/model.pt --device mps --ode-steps 128 --samples-per-class 5

# Select digits
python eval.py runs/<run>/model.pt --device mps --digits 3 7
```

Saves `euler_128.png` and `trajectory_128.png` beside the checkpoint.
Columns are labeled with the requested digit. Each row shares the same initial noise across classes.
Trajectory columns show one shared noise sample becoming each requested digit, with time increasing downwards.
Rerunning eval with the same step count replaces these PNGs.

## Result

3,000 training steps on MPS · 128 Euler steps · sampling seed 2026.
Each column requests one digit, from 0 to 9, with five samples per digit. Some samples still have incomplete strokes.

![Class-conditional MNIST samples](images/conditional.png)

The same initial noise becomes a different digit in each column. Time increases from top to bottom.

![Noise-to-digit trajectories with 128 Euler steps](images/trajectory_128.png)
