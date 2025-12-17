import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

CONFIG = {
    'test_dir': '/kaggle/input/nycu-114-1-introduction-to-machine-learning-hw-4/data/test',
    'model_path': '/kaggle/input/ml-hw4/best_model.pth',
    'output_path': '/kaggle/working/submission.csv',
    
    'model_name': 'convnext_base.fb_in1k',
    'num_classes': 2,
    'drop_rate': 0.15,
    'image_size': 224,
    'batch_size': 8,
    'num_workers': 4,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = np.array(image)
            image = self.transform(image=image)['image']
        
        # Remove file extension from filename (Kaggle requirement)
        filename = Path(img_path).stem  # Changed from .name to .stem
        return image, filename

# ============================================================
# Transform
# ============================================================

def get_test_transform(image_size=224):
    """Test transform without augmentation"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================
# Load Model
# ============================================================

def load_model(model_path, config, device):
    """Load trained model"""
    print(f"\n🔧 Loading model from: {model_path}")
    
    # Create model architecture (must match training)
    model = timm.create_model(
        config['model_name'],
        pretrained=False,
        num_classes=config['num_classes'],
        drop_rate=config['drop_rate']
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'val_acc' in checkpoint:
            print(f"   Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    return model

# ============================================================
# Inference
# ============================================================

def predict(model, test_loader, device):
    """Run inference on test data"""
    model.eval()
    predictions = []
    filenames = []
    
    print("\n🚀 Running inference...")
    with torch.no_grad():
        for images, fnames in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            
            # Get predictions (0: Real, 1: AI-Generated)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            
            predictions.extend(preds)
            filenames.extend(fnames)
    
    return filenames, predictions

# ============================================================
# Generate Submission
# ============================================================

def generate_submission(filenames, predictions, output_path):
    """Create submission CSV file"""
    print(f"\n📄 Generating submission file: {output_path}")
    
    # Create DataFrame with Kaggle required column names
    df = pd.DataFrame({
        'filename': filenames,  # Changed from 'id' to 'filename'
        'label': predictions
    })
    
    # Sort by filename for consistency
    df = df.sort_values('filename').reset_index(drop=True)  # Changed 'id' to 'filename'
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Submission saved!")
    print(f"   Total samples: {len(df)}")
    print(f"   Real (0): {(predictions == 0).sum()}")
    print(f"   AI-Generated (1): {(predictions == 1).sum()}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    
    return df

# ============================================================
# Main
# ============================================================

def main():
    print("="*60)
    print("Quick Inference for Kaggle Submission")
    print("="*60)
    
    # Check if model exists
    model_path = Path(CONFIG['model_path'])
    if not model_path.exists():
        print(f"Model not found: {CONFIG['model_path']}")
        print("\nLooking for model files in /kaggle/working/...")
        working_dir = Path('/kaggle/working/')
        pth_files = list(working_dir.glob('*.pth'))
        if pth_files:
            print(f"Found {len(pth_files)} .pth files:")
            for f in pth_files:
                print(f"   - {f.name}")
            print(f"\n💡 Update CONFIG['model_path'] to use one of these files")
        return
    
    # Load test images
    print(f"\nLoading test images from: {CONFIG['test_dir']}")
    test_dir = Path(CONFIG['test_dir'])
    
    if not test_dir.exists():
        print(f"Test directory not found: {CONFIG['test_dir']}")
        return
    
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    test_image_paths = sorted([
        str(f.absolute()) for f in test_dir.iterdir()
        if f.is_file() and f.suffix.lower() in img_extensions
    ])
    
    print(f"✅ Found {len(test_image_paths)} test images")
    
    if len(test_image_paths) == 0:
        print("No test images found!")
        return
    
    # Create dataset and dataloader
    test_transform = get_test_transform(CONFIG['image_size'])
    test_dataset = TestDataset(test_image_paths, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # Load model
    model = load_model(CONFIG['model_path'], CONFIG, device)
    
    # Run inference
    filenames, predictions = predict(model, test_loader, device)
    
    # Generate submission file
    df = generate_submission(filenames, predictions, CONFIG['output_path'])
    
    print("\n" + "="*60)
    print("✅ Inference completed successfully!")
    print("="*60)
    print(f"\nDownload your submission file:")
    print(f"   {CONFIG['output_path']}")

if __name__ == '__main__':
    main()