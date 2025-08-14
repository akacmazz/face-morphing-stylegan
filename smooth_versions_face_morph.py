import sys
import os

# Add StyleGAN3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan3'))

import pickle
import torch
import cv2
import numpy as np
import PIL.Image
from tqdm import tqdm

def create_smooth_versions(seed1, seed2, base_frames=16):
    """Create smooth and ultra-smooth versions of face morphing"""
    
    model_path = 'stylegan3-t-ffhq-1024x1024.pkl'
    print(f"Loading StyleGAN3 model...")
    
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = G.to(device)
    G.eval()
    
    for param in G.parameters():
        param.requires_grad = False
    
    print(f"Using device: {device}")
    print(f"Creating smooth versions: seed {seed1} → seed {seed2}")
    
    # Generate latent vectors
    torch.manual_seed(seed1)
    z1 = torch.randn([1, G.z_dim], device=device)
    
    torch.manual_seed(seed2)
    z2 = torch.randn([1, G.z_dim], device=device)
    
    # Create base interpolation steps
    alphas = np.linspace(0, 1, base_frames)
    
    print(f"Generating {base_frames} base frames...")
    
    base_frames_data = []
    
    with torch.no_grad():
        for i, alpha in enumerate(tqdm(alphas, desc="Generating base frames")):
            # Linear interpolation
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Generate image
            img = G(z_interp, None)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_array = img[0].cpu().numpy()
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            base_frames_data.append(img_bgr)
    
    # Create different smooth versions
    versions = [
        {
            'name': 'final_face_morph_smooth',
            'description': 'Smooth version (4x frame multiplication)',
            'multiplier': 4,
            'fps': 24,
            'add_interpolation': True
        },
        {
            'name': 'final_face_morph_ultra_smooth',
            'description': 'Ultra-smooth version (6x frame multiplication)',
            'multiplier': 6,
            'fps': 30,
            'add_interpolation': True
        }
    ]
    
    for version in versions:
        print(f"\nCreating {version['name']}: {version['description']}")
        
        # Create video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (1024, 1024)
        video_filename = f"{version['name']}.mp4"
        video_writer = cv2.VideoWriter(video_filename, fourcc, version['fps'], frame_size)
        
        print("Adding forward frames...")
        
        # Forward direction with frame multiplication and interpolation
        for i in range(len(base_frames_data) - 1):
            current_frame = base_frames_data[i]
            next_frame = base_frames_data[i + 1]
            
            # Add current frame multiple times
            for _ in range(version['multiplier']):
                video_writer.write(current_frame)
            
            # Add interpolated frames between current and next
            if version['add_interpolation']:
                for j in range(1, version['multiplier']):
                    alpha = j / version['multiplier']
                    # Simple blend between frames
                    blended = cv2.addWeighted(current_frame, 1-alpha, next_frame, alpha, 0)
                    video_writer.write(blended)
        
        # Add final frame
        for _ in range(version['multiplier']):
            video_writer.write(base_frames_data[-1])
        
        print("Adding backward frames for seamless loop...")
        
        # Backward direction for seamless loop
        reverse_frames = base_frames_data[1:-1]  # Skip first and last to avoid duplicates
        reverse_frames.reverse()
        
        for i in range(len(reverse_frames) - 1):
            current_frame = reverse_frames[i]
            next_frame = reverse_frames[i + 1]
            
            # Add current frame multiple times
            for _ in range(version['multiplier']):
                video_writer.write(current_frame)
            
            # Add interpolated frames
            if version['add_interpolation']:
                for j in range(1, version['multiplier']):
                    alpha = j / version['multiplier']
                    blended = cv2.addWeighted(current_frame, 1-alpha, next_frame, alpha, 0)
                    video_writer.write(blended)
        
        # Add final reverse frame
        if reverse_frames:
            for _ in range(version['multiplier']):
                video_writer.write(reverse_frames[-1])
        
        video_writer.release()
        print(f"✅ Video saved: {video_filename}")
        
        # Create GIF version for this smooth level
        print(f"Creating GIF version...")
        gif_frames = []
        
        # Use every few frames for GIF to keep file size reasonable
        frame_step = max(1, version['multiplier'] // 2)
        
        for i in range(0, len(base_frames_data), max(1, len(base_frames_data) // 12)):
            frame_bgr = base_frames_data[i]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(frame_rgb)
            pil_img = pil_img.resize((512, 512), PIL.Image.Resampling.LANCZOS)
            
            # Add frame multiple times for smoothness
            for _ in range(2):
                gif_frames.append(pil_img)
        
        # Add reverse frames for loop
        reverse_gif_frames = gif_frames[1:-1]
        reverse_gif_frames.reverse()
        all_gif_frames = gif_frames + reverse_gif_frames
        
        gif_filename = f"{version['name']}.gif"
        duration = 150 if 'ultra' in version['name'] else 200
        
        all_gif_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=all_gif_frames[1:],
            duration=duration,
            loop=0
        )
        
        print(f"✅ GIF saved: {gif_filename}")

if __name__ == '__main__':
    print("=== Creating Smooth and Ultra-Smooth Face Morph Versions ===")
    
    # Create smooth versions using the same good quality seeds
    create_smooth_versions(
        seed1=1337,  # Good male face with glasses
        seed2=777,   # Good female face
        base_frames=16
    )
    
    print(f"\n🎉 Smooth versions completed!")
    print(f"\nGenerated files:")
    print(f"📹 final_face_morph_smooth.mp4: Smooth version (4x multiplication, 24 FPS)")
    print(f"🎬 final_face_morph_ultra_smooth.mp4: Ultra-smooth version (6x multiplication, 30 FPS)")
    print(f"🖼️  final_face_morph_smooth.gif: Smooth GIF")
    print(f"🖼️  final_face_morph_ultra_smooth.gif: Ultra-smooth GIF")
    print(f"\n✨ Features:")
    print(f"   • Frame multiplication for smoothness")
    print(f"   • Inter-frame interpolation")
    print(f"   • Seamless loop animation")
    print(f"   • Multiple quality levels")