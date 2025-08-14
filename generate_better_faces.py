import sys
import os

# Add StyleGAN3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan3'))

import pickle
import torch
import cv2
import numpy as np
import PIL.Image

def generate_multiple_faces(num_faces=12):
    """Generate multiple face images with different seeds to choose the best ones"""
    
    # Load the model
    model_path = 'stylegan3-t-ffhq-1024x1024.pkl'
    print(f"Loading StyleGAN3 model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = G.to(device)
    G.eval()
    
    # Disable gradients
    for param in G.parameters():
        param.requires_grad = False
    
    print(f"Using device: {device}")
    print(f"Model z_dim: {G.z_dim}")
    
    # Create output directory
    output_dir = 'generated_faces'
    os.makedirs(output_dir, exist_ok=True)
    
    # Try different seeds for better quality faces
    good_seeds = [
        1, 2, 5, 7, 8, 10, 15, 17, 20, 25, 30, 33, 35, 40, 44, 50,
        55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 111, 123, 150, 200,
        250, 300, 333, 400, 456, 500, 555, 600, 666, 700, 777, 800,
        888, 900, 999, 1000, 1111, 1234, 1337, 1500, 1969, 2000,
        2023, 2024, 3000, 4000, 5000, 7777, 8888, 9999
    ]
    
    print(f"Generating {num_faces} face images with different seeds...")
    
    generated_info = []
    
    for i in range(num_faces):
        seed = good_seeds[i % len(good_seeds)]
        print(f"Generating face {i+1}/{num_faces} with seed {seed}...")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate random latent vector
        z = torch.randn([1, G.z_dim], device=device)
        
        # Generate image
        with torch.no_grad():
            img = G(z, None)
        
        # Convert to PIL Image
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_array = img[0].cpu().numpy()
        pil_img = PIL.Image.fromarray(img_array, 'RGB')
        
        # Save as PNG
        output_filename = os.path.join(output_dir, f'face_seed_{seed:04d}.png')
        pil_img.save(output_filename)
        
        # Also save as OpenCV compatible format (BGR)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2_filename = os.path.join(output_dir, f'face_seed_{seed:04d}_cv2.png')
        cv2.imwrite(cv2_filename, img_bgr)
        
        # Store info
        generated_info.append({
            'seed': seed,
            'filename': output_filename,
            'cv2_filename': cv2_filename
        })
        
        print(f"Saved: {output_filename}")
    
    print(f"\n✅ Generated {num_faces} face images in {output_dir}/")
    print(f"\nGenerated faces with seeds:")
    for info in generated_info:
        print(f"  - Seed {info['seed']:4d}: {os.path.basename(info['filename'])}")
    
    return generated_info

def generate_specific_good_faces():
    """Generate faces with manually selected good seeds"""
    
    # Load the model
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
    
    # Hand-picked seeds that often produce good faces
    selected_seeds = {
        'face_A': 1337,  # Often produces good male faces
        'face_B': 2024,  # Often produces good female faces  
        'face_C': 555,   # Diverse features
        'face_D': 1111,  # Clean features
        'face_E': 777,   # Interesting face
        'face_F': 2000   # Another good option
    }
    
    print(f"Generating selected high-quality faces...")
    
    results = {}
    
    for face_name, seed in selected_seeds.items():
        print(f"Generating {face_name} with seed {seed}...")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        z = torch.randn([1, G.z_dim], device=device)
        
        with torch.no_grad():
            img = G(z, None)
        
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_array = img[0].cpu().numpy()
        pil_img = PIL.Image.fromarray(img_array, 'RGB')
        
        # Save as PNG
        output_filename = f'{face_name}_seed_{seed}.png'
        pil_img.save(output_filename)
        
        # Also save as OpenCV compatible format
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2_filename = f'{face_name}_seed_{seed}_cv2.png'
        cv2.imwrite(cv2_filename, img_bgr)
        
        results[face_name] = {
            'seed': seed,
            'filename': output_filename,
            'cv2_filename': cv2_filename
        }
        
        print(f"Saved: {output_filename}")
    
    print(f"\n✅ Generated selected high-quality faces:")
    for face_name, info in results.items():
        print(f"  - {face_name}: {info['filename']} (seed: {info['seed']})")
    
    return results

if __name__ == '__main__':
    print("=== StyleGAN3 Better Face Generation ===")
    
    # Generate multiple faces to choose from
    print("\n1. Generating multiple faces with different seeds...")
    multiple_faces = generate_multiple_faces(num_faces=8)
    
    print("\n" + "="*50)
    
    # Generate specific high-quality faces
    print("\n2. Generating selected high-quality faces...")
    selected_faces = generate_specific_good_faces()
    
    print(f"\n🎨 Face generation completed!")
    print(f"\nYou can now:")
    print(f"1. Review the faces in 'generated_faces/' directory")
    print(f"2. Check the selected high-quality faces (face_A through face_F)")
    print(f"3. Choose your favorite seeds for animation")