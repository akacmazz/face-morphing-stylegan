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

def slerp(z1, z2, t):
    """Spherical linear interpolation"""
    z1_norm = z1 / torch.norm(z1, dim=1, keepdim=True)
    z2_norm = z2 / torch.norm(z2, dim=1, keepdim=True)
    
    dot_product = torch.sum(z1_norm * z2_norm, dim=1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1, 1)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)
    
    if torch.abs(sin_omega) < 1e-7:
        return (1 - t) * z1 + t * z2
    
    return (torch.sin((1 - t) * omega) / sin_omega) * z1 + (torch.sin(t * omega) / sin_omega) * z2

def ease_in_out_cubic(t):
    """Cubic easing function"""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def generate_fast_smooth_animation(seed1=42, seed2=123):
    """Generate smooth animation quickly with minimal frames but maximum interpolation"""
    
    model_path = 'stylegan3-t-ffhq-1024x1024.pkl'
    print(f"Loading StyleGAN3 model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = G.to(device)
    G.eval()
    
    for param in G.parameters():
        param.requires_grad = False
    
    print(f"Using device: {device}")
    
    # Generate latent vectors
    torch.manual_seed(seed1)
    np.random.seed(seed1)
    z1 = torch.randn([1, G.z_dim], device=device)
    
    torch.manual_seed(seed2)
    np.random.seed(seed2)
    z2 = torch.randn([1, G.z_dim], device=device)
    
    # Generate only 12 key frames but with high quality interpolation
    num_keyframes = 12
    print(f"Generating {num_keyframes} key frames...")
    
    frames_dir = 'fast_smooth_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create eased interpolation points
    linear_alphas = np.linspace(0, 1, num_keyframes)
    eased_alphas = [ease_in_out_cubic(alpha) for alpha in linear_alphas]
    
    key_frames = []
    
    with torch.no_grad():
        for i, alpha in enumerate(tqdm(eased_alphas, desc="Generating key frames")):
            # Use SLERP for smooth interpolation
            z_interp = slerp(z1, z2, alpha)
            
            # Generate image
            img = G(z_interp, None)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_array = img[0].cpu().numpy()
            
            # Save key frame
            frame_filename = os.path.join(frames_dir, f'key_frame_{i:02d}_alpha_{alpha:.3f}.png')
            PIL.Image.fromarray(img_array, 'RGB').save(frame_filename)
            
            # Store for video creation
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            key_frames.append(img_bgr)
    
    print("Creating interpolated smooth videos...")
    
    # Create multiple versions with different interpolation techniques
    video_configs = [
        {
            'filename': 'smooth_face_basic.mp4',
            'description': 'Basic frame multiplication (8x)',
            'multiplier': 8,
            'fps': 24,
            'add_interpolation': False
        },
        {
            'filename': 'smooth_face_interpolated.mp4', 
            'description': 'With frame interpolation',
            'multiplier': 4,
            'fps': 30,
            'add_interpolation': True
        }
    ]
    
    for config in video_configs:
        print(f"Creating {config['filename']}: {config['description']}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (1024, 1024)
        video_writer = cv2.VideoWriter(config['filename'], fourcc, config['fps'], frame_size)
        
        # Forward direction
        if config['add_interpolation']:
            # Create interpolated frames between key frames
            for i in range(len(key_frames) - 1):
                current_frame = key_frames[i]
                next_frame = key_frames[i + 1]
                
                # Add current frame
                for _ in range(config['multiplier']):
                    video_writer.write(current_frame)
                
                # Add interpolated frames (simple blend)
                for j in range(1, config['multiplier']):
                    alpha = j / config['multiplier']
                    blended = cv2.addWeighted(current_frame, 1-alpha, next_frame, alpha, 0)
                    video_writer.write(blended)
            
            # Add final frame
            for _ in range(config['multiplier']):
                video_writer.write(key_frames[-1])
        else:
            # Simple frame multiplication
            for frame in key_frames:
                for _ in range(config['multiplier']):
                    video_writer.write(frame)
        
        # Backward direction for seamless loop
        reverse_frames = key_frames[1:-1]  # Skip first and last to avoid duplicates
        reverse_frames.reverse()
        
        if config['add_interpolation']:
            for i in range(len(reverse_frames) - 1):
                current_frame = reverse_frames[i]
                next_frame = reverse_frames[i + 1]
                
                for _ in range(config['multiplier']):
                    video_writer.write(current_frame)
                
                for j in range(1, config['multiplier']):
                    alpha = j / config['multiplier']
                    blended = cv2.addWeighted(current_frame, 1-alpha, next_frame, alpha, 0)
                    video_writer.write(blended)
            
            if reverse_frames:
                for _ in range(config['multiplier']):
                    video_writer.write(reverse_frames[-1])
        else:
            for frame in reverse_frames:
                for _ in range(config['multiplier']):
                    video_writer.write(frame)
        
        video_writer.release()
        print(f"Saved: {config['filename']}")
    
    # Create high-quality GIF
    print("Creating high-quality GIF...")
    gif_frames = []
    
    # Resize frames for GIF and add interpolation
    for i in range(len(key_frames)):
        frame_bgr = key_frames[i]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(frame_rgb)
        pil_img = pil_img.resize((512, 512), PIL.Image.Resampling.LANCZOS)
        
        # Add frame multiple times for smoothness
        for _ in range(3):
            gif_frames.append(pil_img)
        
        # Add interpolated frame if not the last frame
        if i < len(key_frames) - 1:
            next_frame_bgr = key_frames[i + 1]
            next_frame_rgb = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2RGB)
            next_pil_img = PIL.Image.fromarray(next_frame_rgb)
            next_pil_img = next_pil_img.resize((512, 512), PIL.Image.Resampling.LANCZOS)
            
            # Simple blend for intermediate frame
            blended_array = np.array(pil_img) * 0.5 + np.array(next_pil_img) * 0.5
            blended_img = PIL.Image.fromarray(blended_array.astype(np.uint8))
            gif_frames.append(blended_img)
    
    # Add reverse frames
    reverse_gif_frames = gif_frames[1:-1]
    reverse_gif_frames.reverse()
    all_gif_frames = gif_frames + reverse_gif_frames
    
    gif_filename = 'ultra_smooth_face_morph.gif'
    all_gif_frames[0].save(
        gif_filename,
        save_all=True,
        append_images=all_gif_frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    
    print(f"High-quality GIF saved as: {gif_filename}")
    
    return frames_dir

if __name__ == '__main__':
    print("=== Fast Smooth StyleGAN Face Morphing Animation ===")
    
    frames_dir = generate_fast_smooth_animation(seed1=42, seed2=123)
    
    print(f"\n✅ Animation generation completed!")
    print(f"\nGenerated files:")
    print(f"📹 smooth_face_basic.mp4: Basic smooth animation")
    print(f"🎬 smooth_face_interpolated.mp4: Ultra-smooth with interpolation")
    print(f"🖼️  ultra_smooth_face_morph.gif: High-quality GIF")
    print(f"📁 {frames_dir}/: Key frames")
    
    print(f"\n🔧 Technical features:")
    print(f"   • Spherical interpolation (SLERP)")
    print(f"   • Cubic easing for natural timing")
    print(f"   • Frame interpolation between key frames")
    print(f"   • Seamless loop animation")
    print(f"   • Multiple quality levels")