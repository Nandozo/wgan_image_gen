# WGAN-GP Video Frame Replicator

A Wasserstein GAN with Gradient Penalty that extracts frames from a video and trains to replicate them.

## Features

- **Wasserstein Loss**: More stable training than vanilla GANs
- **Gradient Penalty**: Enforces Lipschitz constraint to prevent mode collapse
- **200x200 Output**: Generates images at 200x200 resolution
- **Frame Extraction**: Automatically extracts every 10th frame from input video
- **Progressive Saving**: Saves sample images every 10 epochs for monitoring

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train from Video

```bash
python wgan_train.py --video front.mp4 --epochs 500 --batch-size 16 --output output
```

### Train from Pre-extracted Frames

```bash
python wgan_train.py --frames-dir /path/to/frames --epochs 500 --output output
```

### Generate Samples from Trained Model

```bash
python wgan_train.py --generate output/checkpoints/final_model.pt --num-samples 32 --output output
```

## Output Structure

```
output/
├── frames/           # Extracted video frames
├── samples/          # Generated samples during training (every 10 epochs)
├── checkpoints/      # Model checkpoints (every 50 epochs + final)
└── generated/        # Generated samples (in generation mode)
```

## Architecture

### Generator
- Input: 128-dimensional latent vector
- Progressive upsampling: 5→10→25→50→100→200
- Batch normalization + LeakyReLU activations
- Tanh output (images in [-1, 1])

### Critic (Discriminator)
- Input: 200x200x3 RGB image
- Progressive downsampling: 200→100→50→25→12→6
- Instance normalization (better for WGAN-GP)
- Linear output (no sigmoid - raw Wasserstein distance)

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Latent Dim | 128 | Size of noise vector |
| Learning Rate | 0.0001 | Adam learning rate |
| Beta1, Beta2 | 0.0, 0.9 | Adam momentum parameters |
| Critic Iterations | 5 | Critic updates per generator update |
| Lambda GP | 10 | Gradient penalty coefficient |
| Batch Size | 16 | Training batch size |

## Tips for Best Results

1. **More data = better**: Extract more frames if possible (reduce interval)
2. **Training duration**: 500+ epochs recommended for good results
3. **Monitor samples**: Check `output/samples/` to track progress
4. **Mode collapse check**: If all samples look identical, increase lambda_gp
5. **GPU recommended**: Training is much faster with CUDA

## Troubleshooting

- **Out of memory**: Reduce batch size
- **Poor quality**: Train for more epochs
- **Mode collapse**: Increase gradient penalty (lambda_gp)
- **Slow training**: Use GPU or reduce image size
