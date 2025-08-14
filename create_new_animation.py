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

def create_new_face_animation(seed1, seed2, animation_name):
    """Create animation between two selected face seeds"""
    
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
    print(f"Creating animation: {animation_name}")
    print(f"From seed {seed1} to seed {seed2}")
    
    # Generate latent vectors
    torch.manual_seed(seed1)
    np.random.seed(seed1)
    z1 = torch.randn([1, G.z_dim], device=device)
    
    torch.manual_seed(seed2)
    np.random.seed(seed2)
    z2 = torch.randn([1, G.z_dim], device=device)
    
    # Generate preview images first
    print("Generating preview images...")
    
    with torch.no_grad():
        # Generate face 1
        img1 = G(z1, None)
        img1 = (img1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img1_array = img1[0].cpu().numpy()
        pil_img1 = PIL.Image.fromarray(img1_array, 'RGB')
        preview1_filename = f'{animation_name}_face1_seed_{seed1}.png'
        pil_img1.save(preview1_filename)
        print(f"Face 1 preview saved: {preview1_filename}")
        
        # Generate face 2
        img2 = G(z2, None)
        img2 = (img2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img2_array = img2[0].cpu().numpy()
        pil_img2 = PIL.Image.fromarray(img2_array, 'RGB')
        preview2_filename = f'{animation_name}_face2_seed_{seed2}.png'
        pil_img2.save(preview2_filename)
        print(f"Face 2 preview saved: {preview2_filename}")
    
    # Generate animation frames
    num_keyframes = 15  # More frames for better quality
    print(f"Generating {num_keyframes} key frames...")
    
    frames_dir = f'{animation_name}_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create eased interpolation points
    linear_alphas = np.linspace(0, 1, num_keyframes)
    eased_alphas = [ease_in_out_cubic(alpha) for alpha in linear_alphas]
    
    key_frames = []
    
    with torch.no_grad():
        for i, alpha in enumerate(tqdm(eased_alphas, desc="Generating animation frames")):
            # Use SLERP for smooth interpolation
            z_interp = slerp(z1, z2, alpha)
            
            # Generate image
            img = G(z_interp, None)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_array = img[0].cpu().numpy()
            
            # Save key frame
            frame_filename = os.path.join(frames_dir, f'frame_{i:02d}_alpha_{alpha:.3f}.png')
            PIL.Image.fromarray(img_array, 'RGB').save(frame_filename)
            
            # Store for video creation
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            key_frames.append(img_bgr)
    
    print("Creating high-quality videos...")
    
    # Create multiple video versions
    video_configs = [
        {
            'filename': f'{animation_name}_smooth.mp4',
            'description': 'Smooth animation (6x multiplication)',
            'multiplier': 6,
            'fps': 30,
            'add_interpolation': True
        },
        {
            'filename': f'{animation_name}_ultra_smooth.mp4', 
            'description': 'Ultra-smooth animation (8x multiplication)',
            'multiplier': 8,
            'fps': 30,
            'add_interpolation': True
        }
    ]
    
    for config in video_configs:
        print(f"Creating {config['filename']}: {config['description']}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (1024, 1024)
        video_writer = cv2.VideoWriter(config['filename'], fourcc, config['fps'], frame_size)
        
        # Forward direction with interpolation
        for i in range(len(key_frames) - 1):
            current_frame = key_frames[i]
            next_frame = key_frames[i + 1]
            
            # Add current frame multiple times
            for _ in range(config['multiplier']):
                video_writer.write(current_frame)
            
            # Add interpolated frames
            if config['add_interpolation']:
                for j in range(1, config['multiplier']):
                    alpha = j / config['multiplier']
                    blended = cv2.addWeighted(current_frame, 1-alpha, next_frame, alpha, 0)
                    video_writer.write(blended)
        
        # Add final frame
        for _ in range(config['multiplier']):
            video_writer.write(key_frames[-1])
        
        # Backward direction for seamless loop
        reverse_frames = key_frames[1:-1]
        reverse_frames.reverse()
        
        for i in range(len(reverse_frames) - 1):
            current_frame = reverse_frames[i]
            next_frame = reverse_frames[i + 1]
            
            for _ in range(config['multiplier']):
                video_writer.write(current_frame)
            
            if config['add_interpolation']:
                for j in range(1, config['multiplier']):
                    alpha = j / config['multiplier']
                    blended = cv2.addWeighted(current_frame, 1-alpha, next_frame, alpha, 0)
                    video_writer.write(blended)
        
        if reverse_frames:
            for _ in range(config['multiplier']):
                video_writer.write(reverse_frames[-1])
        
        video_writer.release()
        print(f"✅ Saved: {config['filename']}")
    
    # Create GIF version
    print("Creating high-quality GIF...")
    gif_frames = []
    
    for i in range(len(key_frames)):
        frame_bgr = key_frames[i]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(frame_rgb)
        pil_img = pil_img.resize((512, 512), PIL.Image.Resampling.LANCZOS)
        
        # Add frame multiple times for smoothness
        for _ in range(4):
            gif_frames.append(pil_img)
        
        # Add interpolated frame if not the last frame
        if i < len(key_frames) - 1:
            next_frame_bgr = key_frames[i + 1]
            next_frame_rgb = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2RGB)
            next_pil_img = PIL.Image.fromarray(next_frame_rgb)
            next_pil_img = next_pil_img.resize((512, 512), PIL.Image.Resampling.LANCZOS)
            
            # Blend for intermediate frame
            blended_array = np.array(pil_img) * 0.5 + np.array(next_pil_img) * 0.5
            blended_img = PIL.Image.fromarray(blended_array.astype(np.uint8))
            gif_frames.append(blended_img)
    
    # Add reverse frames
    reverse_gif_frames = gif_frames[1:-1]
    reverse_gif_frames.reverse()
    all_gif_frames = gif_frames + reverse_gif_frames
    
    gif_filename = f'{animation_name}.gif'
    all_gif_frames[0].save(
        gif_filename,
        save_all=True,
        append_images=all_gif_frames[1:],
        duration=80,  # 80ms per frame for very smooth animation
        loop=0
    )
    
    print(f"✅ High-quality GIF saved: {gif_filename}")
    
    return frames_dir

if __name__ == '__main__':
    print("=== Creating New High-Quality Face Morphing Animation ===")
    
    # Create animation with the best faces
    print("\n🎬 Creating animation: Face A (1337) → Face E (777)")
    create_new_face_animation(
        seed1=1337,  # Face A - good male face with glasses
        seed2=777,   # Face E - good female face 
        animation_name='best_face_morph_A_to_E'
    )
    
    print("\n" + "="*60)
    
    # Create another animation with different combination
    print("\n🎬 Creating animation: Face D (1111) → Face B (2024)")
    create_new_face_animation(
        seed1=1111,  # Face D - clean features
        seed2=2024,  # Face B - another good face
        animation_name='best_face_morph_D_to_B'
    )
    
    print(f"\n🎉 High-quality animations completed!")
    print(f"\nGenerated files:")
    print(f"📹 best_face_morph_A_to_E_smooth.mp4")
    print(f"🎬 best_face_morph_A_to_E_ultra_smooth.mp4")
    print(f"🖼️  best_face_morph_A_to_E.gif")
    print(f"📹 best_face_morph_D_to_B_smooth.mp4")
    print(f"🎬 best_face_morph_D_to_B_ultra_smooth.mp4")
    print(f"🖼️  best_face_morph_D_to_B.gif")
    print(f"\n✨ These animations use much better quality faces!")