#!/usr/bin/env python3
"""
WGAN-GP (Wasserstein GAN with Gradient Penalty) for Video Frame Replication
Extracts every 10th frame from a video and trains a GAN to replicate them.

Key features:
- Wasserstein loss for stable training
- Gradient penalty to enforce Lipschitz constraint (prevents mode collapse)
- Progressive saving of generated samples
- 200x200 image size
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Image settings
    IMAGE_SIZE = 200
    CHANNELS = 3
    
    # Model settings
    LATENT_DIM = 128
    GEN_FEATURES = 64
    CRITIC_FEATURES = 64
    
    # Training settings
    BATCH_SIZE = 16
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.0001
    BETA1 = 0.0
    BETA2 = 0.9
    
    # WGAN-GP settings
    CRITIC_ITERATIONS = 5  # Train critic more than generator
    LAMBDA_GP = 10  # Gradient penalty coefficient
    
    # Frame extraction
    FRAME_INTERVAL = 10  # Extract every 10th frame
    
    # Paths
    OUTPUT_DIR = "output"
    FRAMES_DIR = "frames"
    SAMPLES_DIR = "samples"
    CHECKPOINTS_DIR = "checkpoints"


# ============================================================================
# Frame Extraction
# ============================================================================
def extract_frames(video_path: str, output_dir: str, interval: int = 10) -> int:
    """Extract every nth frame from a video file."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    print(f"Extracting every {interval}th frame...")
    
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames // interval, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Save frame
                save_path = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
                img.save(save_path)
                saved_count += 1
                pbar.update(1)
            
            frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return saved_count


# ============================================================================
# Dataset
# ============================================================================
class FrameDataset(Dataset):
    """Dataset for loading extracted video frames."""
    
    def __init__(self, frames_dir: str, image_size: int = 200):
        self.frames_dir = frames_dir
        self.image_files = sorted([
            f for f in os.listdir(frames_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {frames_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.frames_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


# ============================================================================
# Generator Network
# ============================================================================
class Generator(nn.Module):
    """
    Generator network using transposed convolutions.
    Takes a latent vector and generates a 200x200 RGB image.
    """
    
    def __init__(self, latent_dim: int, channels: int, features: int):
        super().__init__()
        
        # Calculate initial size for upsampling to 200x200
        # We'll use a series of upsampling layers
        # Starting from 5x5 -> 10 -> 25 -> 50 -> 100 -> 200
        
        self.init_size = 5
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, features * 16 * self.init_size * self.init_size)
        )
        
        self.conv_blocks = nn.Sequential(
            # 5x5 -> 10x10
            nn.BatchNorm2d(features * 16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features * 16, features * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 10x10 -> 25x25
            nn.Upsample(size=(25, 25)),
            nn.Conv2d(features * 8, features * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 25x25 -> 50x50
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features * 4, features * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 50x50 -> 100x100
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features * 2, features, 3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 100x100 -> 200x200
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features, channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# ============================================================================
# Critic (Discriminator) Network
# ============================================================================
class Critic(nn.Module):
    """
    Critic network (discriminator in WGAN terminology).
    Takes a 200x200 RGB image and outputs a scalar score.
    No sigmoid - outputs raw Wasserstein distance estimate.
    """
    
    def __init__(self, channels: int, features: int):
        super().__init__()
        
        self.model = nn.Sequential(
            # 200x200 -> 100x100
            nn.Conv2d(channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 100x100 -> 50x50
            nn.Conv2d(features, features * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 50x50 -> 25x25
            nn.Conv2d(features * 2, features * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 25x25 -> 12x12
            nn.Conv2d(features * 4, features * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(features * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 12x12 -> 6x6
            nn.Conv2d(features * 8, features * 16, 4, stride=2, padding=1),
            nn.InstanceNorm2d(features * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate output size: 200 -> 100 -> 50 -> 25 -> 12 -> 6
        self.fc = nn.Linear(features * 16 * 6 * 6, 1)
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.fc(out)
        return validity


# ============================================================================
# Gradient Penalty
# ============================================================================
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP.
    Enforces Lipschitz constraint for stable training.
    """
    batch_size = real_samples.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    # Interpolate between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Get critic scores for interpolated samples
    d_interpolated = critic(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


# ============================================================================
# Training Loop
# ============================================================================
def train_wgan(
    generator: nn.Module,
    critic: nn.Module,
    dataloader: DataLoader,
    config: Config,
    device: torch.device,
    output_dir: str
):
    """Main training loop for WGAN-GP."""
    
    # Create output directories
    samples_dir = os.path.join(output_dir, config.SAMPLES_DIR)
    checkpoints_dir = os.path.join(output_dir, config.CHECKPOINTS_DIR)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Optimizers (Adam with specific betas for WGAN)
    opt_g = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    opt_c = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(16, config.LATENT_DIM, device=device)
    
    # Training history
    g_losses = []
    c_losses = []
    
    print(f"\nStarting WGAN-GP training for {config.NUM_EPOCHS} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Critic iterations per generator iteration: {config.CRITIC_ITERATIONS}")
    print(f"Gradient penalty lambda: {config.LAMBDA_GP}\n")
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_g_loss = 0
        epoch_c_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for i, real_imgs in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # ---------------------
            # Train Critic
            # ---------------------
            for _ in range(config.CRITIC_ITERATIONS):
                opt_c.zero_grad()
                
                # Generate fake images
                z = torch.randn(batch_size, config.LATENT_DIM, device=device)
                fake_imgs = generator(z).detach()
                
                # Critic scores
                real_validity = critic(real_imgs)
                fake_validity = critic(fake_imgs)
                
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs, device)
                
                # Wasserstein loss + gradient penalty
                c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + config.LAMBDA_GP * gradient_penalty
                
                c_loss.backward()
                opt_c.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            opt_g.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, config.LATENT_DIM, device=device)
            fake_imgs = generator(z)
            
            # Generator loss (wants critic to think fakes are real)
            g_loss = -torch.mean(critic(fake_imgs))
            
            g_loss.backward()
            opt_g.step()
            
            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_c_loss += c_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'C_loss': f'{c_loss.item():.4f}'
            })
        
        # Average losses for epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_c_loss = epoch_c_loss / num_batches
        g_losses.append(avg_g_loss)
        c_losses.append(avg_c_loss)
        
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
              f"G_loss: {avg_g_loss:.4f}, C_loss: {avg_c_loss:.4f}")
        
        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            generator.eval()
            with torch.no_grad():
                fake_samples = generator(fixed_noise)
                # Denormalize from [-1, 1] to [0, 1]
                fake_samples = (fake_samples + 1) / 2
                save_image(fake_samples, 
                          os.path.join(samples_dir, f"epoch_{epoch+1:04d}.png"),
                          nrow=4, normalize=False)
            generator.train()
        
        # Save checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'opt_c_state_dict': opt_c.state_dict(),
                'g_losses': g_losses,
                'c_losses': c_losses,
            }, os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Save final model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'config': vars(config),
    }, os.path.join(checkpoints_dir, "final_model.pt"))
    
    print(f"\nTraining complete! Final model saved to {checkpoints_dir}/final_model.pt")
    
    return g_losses, c_losses


# ============================================================================
# Generate Samples from Trained Model
# ============================================================================
def generate_samples(model_path: str, num_samples: int, output_dir: str, device: torch.device):
    """Generate samples from a trained model."""
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = Config()
    generator = Generator(config.LATENT_DIM, config.CHANNELS, config.GEN_FEATURES).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            z = torch.randn(1, config.LATENT_DIM, device=device)
            fake_img = generator(z)
            fake_img = (fake_img + 1) / 2  # Denormalize
            save_image(fake_img, os.path.join(output_dir, f"generated_{i:04d}.png"))
    
    print(f"Samples saved to {output_dir}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="WGAN-GP for Video Frame Replication")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--frames-dir", type=str, help="Path to pre-extracted frames (skip extraction)")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--generate", type=str, help="Path to trained model for generation mode")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generation mode
    if args.generate:
        generate_samples(args.generate, args.num_samples, 
                        os.path.join(args.output, "generated"), device)
        return
    
    # Training mode
    config = Config()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.OUTPUT_DIR = args.output
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Extract frames or use existing
    frames_dir = os.path.join(args.output, config.FRAMES_DIR)
    
    if args.frames_dir:
        frames_dir = args.frames_dir
        print(f"Using pre-extracted frames from {frames_dir}")
    elif args.video:
        extract_frames(args.video, frames_dir, config.FRAME_INTERVAL)
    else:
        # Check if frames already exist
        if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
            print("Error: Please provide --video or --frames-dir argument")
            sys.exit(1)
        print(f"Using existing frames from {frames_dir}")
    
    # Create dataset and dataloader
    dataset = FrameDataset(frames_dir, config.IMAGE_SIZE)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset: {len(dataset)} images")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Initialize models
    generator = Generator(config.LATENT_DIM, config.CHANNELS, config.GEN_FEATURES).to(device)
    critic = Critic(config.CHANNELS, config.CRITIC_FEATURES).to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    c_params = sum(p.numel() for p in critic.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Critic parameters: {c_params:,}")
    
    # Train
    train_wgan(generator, critic, dataloader, config, device, args.output)


if __name__ == "__main__":
    main()
