# Scripts

This folder holds helper scripts for working with the Luogu captcha CRNN.

## Available Scripts

- `smoke_model.py`: Quick sanity check on forward pass and decoding with random input.
- `train.py`: Full training loop with CTC loss, checkpointing, and resume support.
  - Saves `crnn.pt` (latest) every epoch
  - Saves `crnn_best.pt` (best) when validation accuracy improves
  - Supports resuming from checkpoint with `--resume`
- `predict.py`: Fetch captchas from Luogu endpoint, predict, and display with matplotlib.

## Training Workflow

**Generate data:**

```bash
php generate.php 50000
```

**Train from scratch (30 epochs, batch 128):**

```bash
python scripts/train.py --epochs 30 --batch-size 128 --weight-decay 1e-4
```

**Resume from best checkpoint (10 more epochs):**

```bash
python scripts/train.py --epochs 10 --batch-size 128 --resume checkpoints/crnn_best.pt
```

**Live inference:**

```bash
python scripts/predict.py --ckpt checkpoints/crnn_best.pt --count 3
```

## Quick Testing

Run smoke test:

```bash
python scripts/smoke_model.py
```

Run pytest:

```bash
pytest -q
```

Feel free to extend these scripts with additional training, evaluation, and inference utilities.
