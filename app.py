#!/usr/bin/env python3
"""
WGAN-GP Streamlit Demo App
Interactive frontend for exploring the trained WGAN model.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path
import io

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="WGAN-GP Frame Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Model Architecture (must match training)
# ============================================================================
class Generator(nn.Module):
    """Generator network - must match the training architecture exactly."""
    
    def __init__(self, latent_dim: int = 128, channels: int = 3, features: int = 64):
        super().__init__()
        
        self.init_size = 5
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, features * 16 * self.init_size * self.init_size)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(features * 16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features * 16, features * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(size=(25, 25)),
            nn.Conv2d(features * 8, features * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features * 4, features * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features * 2, features, 3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        self.latent_dim = latent_dim
        self.features = features
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# ============================================================================
# Helper Functions
# ============================================================================
@st.cache_resource
def load_model(model_path: str):
    """Load the trained generator model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    generator = Generator(latent_dim=128, channels=3, features=64).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    return generator, device


def tensor_to_image(tensor):
    """Convert a tensor to PIL Image."""
    # Denormalize from [-1, 1] to [0, 1]
    img = (tensor + 1) / 2
    img = img.clamp(0, 1)
    
    # Convert to numpy
    img = img.cpu().detach().numpy()
    
    # Handle batch dimension
    if len(img.shape) == 4:
        img = img[0]
    
    # CHW -> HWC
    img = np.transpose(img, (1, 2, 0))
    
    # Convert to uint8
    img = (img * 255).astype(np.uint8)
    
    return Image.fromarray(img)


def generate_images(generator, device, num_images: int, seed: int = None):
    """Generate images from random latent vectors."""
    if seed is not None:
        torch.manual_seed(seed)
    
    with torch.no_grad():
        z = torch.randn(num_images, 128, device=device)
        fake_imgs = generator(z)
    
    return [tensor_to_image(fake_imgs[i]) for i in range(num_images)]


def interpolate_latent(generator, device, z1, z2, steps: int = 10):
    """Interpolate between two latent vectors."""
    images = []
    
    with torch.no_grad():
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            img = generator(z)
            images.append(tensor_to_image(img))
    
    return images


def load_training_samples(samples_dir: str):
    """Load training progression samples."""
    sample_files = sorted(glob.glob(os.path.join(samples_dir, "epoch_*.png")))
    return sample_files


def load_real_frames(frames_dir: str, max_frames: int = 10):
    """Load real extracted frames for comparison."""
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))[:max_frames]
    return frame_files


# ============================================================================
# Streamlit App
# ============================================================================
def main():
    # Sidebar
    st.sidebar.title("🎬 WGAN-GP Demo")
    st.sidebar.markdown("---")
    
    # Model path configuration
    default_model_path = "output/checkpoints/final_model.pt"
    model_path = st.sidebar.text_input("Model Path", value=default_model_path)
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at `{model_path}`")
        st.info("Please ensure training has completed and the model path is correct.")
        
        st.markdown("### Expected file structure:")
        st.code("""
output/
├── checkpoints/
│   └── final_model.pt  ← Model file
├── samples/
│   └── epoch_*.png     ← Training samples
└── frames/
    └── frame_*.png     ← Extracted frames
        """)
        return
    
    # Load model
    try:
        generator, device = load_model(model_path)
        st.sidebar.success(f"✅ Model loaded ({device})")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["🎲 Generate Images", "🔄 Latent Interpolation", "📊 Training Progress", "⚖️ Real vs Generated", "📓 Notebook"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This demo showcases a **WGAN-GP** (Wasserstein GAN with Gradient Penalty) 
    trained on video frames.
    
    The model learned to generate new images that resemble the training data.
    """)
    
    # ========================================================================
    # Page: Generate Images
    # ========================================================================
    if page == "🎲 Generate Images":
        st.title("🎲 Generate New Images")
        st.markdown("Generate novel images from random latent vectors.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            num_images = st.slider("Number of images", 1, 16, 4)
            seed = st.number_input("Random seed (optional)", min_value=0, max_value=99999, value=42)
            use_seed = st.checkbox("Use fixed seed", value=False)
            
            if st.button("🎨 Generate!", type="primary", use_container_width=True):
                with st.spinner("Generating..."):
                    images = generate_images(
                        generator, device, num_images, 
                        seed=seed if use_seed else None
                    )
                    st.session_state['generated_images'] = images
        
        with col2:
            if 'generated_images' in st.session_state:
                images = st.session_state['generated_images']
                
                # Display in grid
                cols = st.columns(min(4, len(images)))
                for idx, img in enumerate(images):
                    with cols[idx % 4]:
                        st.image(img, caption=f"Generated #{idx+1}", use_container_width=True)
                
                # Download button for all images
                if len(images) > 0:
                    st.markdown("---")
                    
                    # Create download for first image
                    buf = io.BytesIO()
                    images[0].save(buf, format='PNG')
                    st.download_button(
                        label="📥 Download First Image",
                        data=buf.getvalue(),
                        file_name="generated_image.png",
                        mime="image/png"
                    )
    
    # ========================================================================
    # Page: Latent Interpolation
    # ========================================================================
    elif page == "🔄 Latent Interpolation":
        st.title("🔄 Latent Space Interpolation")
        st.markdown("Smoothly morph between two generated images by interpolating in latent space.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            steps = st.slider("Interpolation steps", 5, 20, 10)
            seed1 = st.number_input("Seed for Image A", min_value=0, max_value=99999, value=42)
            seed2 = st.number_input("Seed for Image B", min_value=0, max_value=99999, value=123)
            
            if st.button("🔄 Interpolate!", type="primary", use_container_width=True):
                with st.spinner("Interpolating..."):
                    torch.manual_seed(seed1)
                    z1 = torch.randn(1, 128, device=device)
                    
                    torch.manual_seed(seed2)
                    z2 = torch.randn(1, 128, device=device)
                    
                    images = interpolate_latent(generator, device, z1, z2, steps)
                    st.session_state['interpolated_images'] = images
        
        with col2:
            if 'interpolated_images' in st.session_state:
                images = st.session_state['interpolated_images']
                
                # Show endpoints
                st.markdown("**Image A → Image B**")
                endpoint_cols = st.columns(2)
                with endpoint_cols[0]:
                    st.image(images[0], caption="Start (A)", use_container_width=True)
                with endpoint_cols[1]:
                    st.image(images[-1], caption="End (B)", use_container_width=True)
                
                st.markdown("---")
                st.markdown("**Interpolation Sequence**")
                
                # Slider to scrub through
                frame_idx = st.slider("Frame", 0, len(images)-1, 0)
                st.image(images[frame_idx], caption=f"Step {frame_idx+1}/{len(images)}", width=300)
                
                # Show all frames
                with st.expander("View all frames"):
                    cols = st.columns(5)
                    for idx, img in enumerate(images):
                        with cols[idx % 5]:
                            st.image(img, caption=f"{idx+1}", use_container_width=True)
    
    # ========================================================================
    # Page: Training Progress
    # ========================================================================
    elif page == "📊 Training Progress":
        st.title("📊 Training Progress")
        st.markdown("View how the generator improved over training epochs.")
        
        samples_dir = "output/samples"
        
        if not os.path.exists(samples_dir):
            st.warning(f"Samples directory not found: `{samples_dir}`")
            return
        
        sample_files = load_training_samples(samples_dir)
        
        if not sample_files:
            st.warning("No training samples found.")
            return
        
        st.success(f"Found {len(sample_files)} training snapshots")
        
        # Epoch selector
        epochs = [int(Path(f).stem.split('_')[1]) for f in sample_files]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_epoch = st.select_slider(
                "Select Epoch",
                options=epochs,
                value=epochs[-1]
            )
            
            # Find corresponding file
            idx = epochs.index(selected_epoch)
            
            st.markdown("---")
            st.markdown("### Training Info")
            st.metric("Total Epochs", epochs[-1])
            st.metric("Snapshots Saved", len(sample_files))
        
        with col2:
            st.image(sample_files[idx], caption=f"Epoch {selected_epoch}", use_container_width=True)
        
        # Show progression
        st.markdown("---")
        st.markdown("### Full Training Progression")
        
        # Select subset for display
        display_indices = np.linspace(0, len(sample_files)-1, min(10, len(sample_files)), dtype=int)
        
        cols = st.columns(5)
        for i, idx in enumerate(display_indices):
            with cols[i % 5]:
                st.image(sample_files[idx], caption=f"Epoch {epochs[idx]}", use_container_width=True)
    
    # ========================================================================
    # Page: Real vs Generated
    # ========================================================================
    elif page == "⚖️ Real vs Generated":
        st.title("⚖️ Real vs Generated Comparison")
        st.markdown("Compare original training frames with generated images.")
        
        frames_dir = "output/frames"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Real Frames")
            
            if os.path.exists(frames_dir):
                real_frames = load_real_frames(frames_dir, max_frames=10)

                if real_frames:
                    st.info(f"Showing {len(real_frames)} sample frames (max 10)")
                    frame_cols = st.columns(5)
                    for idx, frame_path in enumerate(real_frames):
                        with frame_cols[idx % 5]:
                            st.image(frame_path, use_container_width=True, caption=f"Frame {idx+1}")
                else:
                    st.warning("No frames found")
            else:
                st.warning(f"Frames directory not found: `{frames_dir}`")
        
        with col2:
            st.markdown("### 🎨 Generated Images")
            
            if st.button("Generate New Batch", use_container_width=True):
                with st.spinner("Generating..."):
                    images = generate_images(generator, device, 8)
                    st.session_state['comparison_images'] = images
            
            if 'comparison_images' in st.session_state:
                images = st.session_state['comparison_images']
                gen_cols = st.columns(4)
                for idx, img in enumerate(images[:8]):
                    with gen_cols[idx % 4]:
                        st.image(img, use_container_width=True)
            else:
                st.info("Click 'Generate New Batch' to create images")

    # ========================================================================
    # Page: Notebook
    # ========================================================================
    elif page == "📓 Notebook":
        st.title("📓 Project Notebook")
        st.markdown("Full technical documentation and implementation details.")

        notebook_path = "RL_image_gen_Notebook.ipynb"

        if not os.path.exists(notebook_path):
            st.error(f"❌ Notebook not found at `{notebook_path}`")
            st.info("Please ensure the notebook file is in the project directory.")
            return

        st.success(f"✅ Notebook found: `{notebook_path}`")

        # Display notebook info
        st.markdown("---")
        st.markdown("### 📖 Viewing Options")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Option 1: Download Notebook**")
            st.markdown("Download the notebook to view in Jupyter or VS Code.")

            # Read notebook file
            with open(notebook_path, 'rb') as f:
                notebook_data = f.read()

            st.download_button(
                label="📥 Download Notebook (.ipynb)",
                data=notebook_data,
                file_name="RL_image_gen_Notebook.ipynb",
                mime="application/x-ipynb+json",
                use_container_width=True
            )

        with col2:
            st.markdown("**Option 2: View on nbviewer**")
            st.markdown("Upload the notebook to GitHub and view it using nbviewer.org or GitHub's viewer.")
            st.markdown("[nbviewer.org](https://nbviewer.org/) - Paste your GitHub notebook URL")

        st.markdown("---")
        st.markdown("### 📋 Notebook Contents")

        # Show notebook structure
        try:
            import json
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)

            # Extract markdown headers
            headers = []
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'markdown':
                    source = ''.join(cell.get('source', []))
                    lines = source.split('\n')
                    for line in lines:
                        if line.startswith('#'):
                            headers.append(line)

            # Display table of contents
            st.markdown("**Table of Contents:**")
            for header in headers[:20]:  # Show first 20 headers
                # Count # symbols for indentation
                level = len(header) - len(header.lstrip('#'))
                title = header.lstrip('#').strip()
                indent = "  " * (level - 1)
                st.markdown(f"{indent}- {title}")

            if len(headers) > 20:
                st.markdown(f"*...and {len(headers) - 20} more sections*")

            st.markdown("---")
            st.info("💡 **Tip:** Download the notebook above to view the complete implementation with code, equations, and visualizations.")

        except Exception as e:
            st.warning(f"Could not parse notebook structure: {e}")


if __name__ == "__main__":
    main()
