"""
Transfer Learning + Semi-Supervised Learning (FixMatch)
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check required libraries
try:
    import timm
    print("Correct: timm 已安裝")
except:
    print("請安裝 timm: pip install timm")
    exit(1)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    print("Correct: albumentations 已安裝")
    USE_ALBUMENTATIONS = True
except:
    print("albumentations 未安裝")
    print("建議安裝: pip install albumentations")
    from torchvision import transforms
    USE_ALBUMENTATIONS = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_dataset(real_dir, generated_dir, unlabeled_dir=None, val_split=0.2, seed=42):
    """
    return:
        train_files, train_labels
        val_files, val_labels
        unlabeled_files
    """
    print("\n" + "="*60)
    print("📁 資料準備中")
    print("="*60)
    
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_files = []
    all_labels = []
    
    print(f"\n掃描真實圖片資料夾: {real_dir}")
    real_path = Path(real_dir)
    if not real_path.exists():
        raise FileNotFoundError(f"找不到資料夾: {real_dir}")
    
    real_images = [f for f in real_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in img_extensions]
    print(f"   ✅ 找到 {len(real_images)} 張真實圖片")
    
    for img in real_images:
        all_files.append(str(img.absolute()))
        all_labels.append(0)  # Real = 0
    
    print(f"\n掃描 AI 生成圖片資料夾: {generated_dir}")
    gen_path = Path(generated_dir)
    if not gen_path.exists():
        raise FileNotFoundError(f"找不到資料夾: {generated_dir}")
    
    gen_images = [f for f in gen_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in img_extensions]
    print(f"   ✅ 找到 {len(gen_images)} 張 AI 生成圖片")
    
    for img in gen_images:
        all_files.append(str(img.absolute()))
        all_labels.append(1)  # AI-Generated = 1
    
    print(f"\n📊 總計: {len(all_files)} 張已標註圖片")
    print(f"   - Real: {sum(1 for l in all_labels if l == 0)}")
    print(f"   - AI-Generated: {sum(1 for l in all_labels if l == 1)}")
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, 
        test_size=val_split, 
        random_state=seed,
        stratify=all_labels
    )
    
    print(f"\n資料分割:")
    print(f"   - 訓練集: {len(train_files)} 張")
    print(f"   - 驗證集: {len(val_files)} 張")
    
    unlabeled_files = []
    if unlabeled_dir:
        print(f"\n掃描無標註資料夾: {unlabeled_dir}")
        unlabeled_path = Path(unlabeled_dir)
        if unlabeled_path.exists():
            unlabeled_files = [str(f.absolute()) for f in unlabeled_path.rglob("*") 
                              if f.is_file() and f.suffix.lower() in img_extensions]
            print(f"   ✅ 找到 {len(unlabeled_files)} 張無標註圖片")
        else:
            print(f"找不到資料夾，跳過無標註資料")
    
    return train_files, train_labels, val_files, val_labels, unlabeled_files


# Data Augmentation
def get_transforms(image_size=224, augment_type='weak'):
    """
    augment_type: 'weak', 'strong', 'test'
    """
    if USE_ALBUMENTATIONS:
        if augment_type == 'weak':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        elif augment_type == 'strong':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(p=0.5),
                ], p=0.3),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
                A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:  # test
            return A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    else:
        # torchvision transforms
        if augment_type == 'weak':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif augment_type == 'strong':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # test
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


class ImageDataset(Dataset):
    def __init__(self, file_list, labels=None, transform=None, use_albu=True):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        self.use_albu = use_albu
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            if self.use_albu:
                image = np.array(image)
                image = self.transform(image=image)['image']
            else:
                image = self.transform(image)
        
        if self.labels is None:
            return image, -1
        else:
            return image, self.labels[idx]


# ============================================================
# 5. EMA (Exponential Moving Average) for Teacher Model
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])



def train_model(
    model,
    train_loader,
    val_loader,
    unlabeled_loader=None,
    epochs=30,
    lr=3e-5,  # Modified from 1e-4 to 3e-5
    device='cuda',
    use_semi_supervised=True,
    threshold=0.98,
    unsup_weight=0.5
):
   
    print("\n" + "="*60)
    print("開始訓練")
    print("="*60)
    print(f"模式: {'半監督學習 (FixMatch)' if use_semi_supervised else '監督式學習'}")
    print(f"裝置: {device}")
    print(f"訓練輪數: {epochs}")
    print(f"學習率: {lr}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)  # 增加 weight decay
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 加入 label smoothing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Check early stopping
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')
    
    # EMA teacher (semi-supervised)
    ema = EMA(model, decay=0.999) if use_semi_supervised else None
    
    # Transform for strong augmentation
    strong_transform = get_transforms(image_size=224, augment_type='strong')
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        sup_loss_sum = 0
        unsup_loss_sum = 0
        correct = 0
        total = 0
        
        # Create iterator if there is unlabeled data
        if use_semi_supervised and unlabeled_loader:
            unlabeled_iter = iter(unlabeled_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Supervised Loss
            outputs = model(images)
            loss_sup = criterion(outputs, labels)
            
            loss = loss_sup
            sup_loss_sum += loss_sup.item()
            
            # Semi-Supervised Loss (FixMatch)
            loss_unsup = torch.tensor(0.0).to(device)
            mask_ratio = 0.0
            
            if use_semi_supervised and unlabeled_loader:
                try:
                    unlabeled_images, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_images, _ = next(unlabeled_iter)
                
                unlabeled_images = unlabeled_images.to(device)
                
                # Create pseudo-labels with weak augmentation
                with torch.no_grad():
                    logits_weak = model(unlabeled_images)
                    probs_weak = torch.softmax(logits_weak, dim=1)
                    max_probs, pseudo_labels = torch.max(probs_weak, dim=1)
                    mask = max_probs.ge(threshold).float()
                    mask_ratio = mask.mean().item()
                
                # Strong augmentation and compute unsupervised loss
                if mask.sum() > 0:
                    strong_images = []
                    for img in unlabeled_images:
                        img_np = img.cpu().permute(1, 2, 0).numpy()
                        # Non-normalize
                        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        if USE_ALBUMENTATIONS:
                            img_strong = strong_transform(image=img_np)['image']
                        else:
                            img_pil = Image.fromarray(img_np)
                            img_strong = strong_transform(img_pil)
                        strong_images.append(img_strong)
                    
                    strong_images = torch.stack(strong_images).to(device)
                    logits_strong = model(strong_images)
                    
                    loss_unsup = (F.cross_entropy(logits_strong, pseudo_labels, reduction='none') * mask).mean()
                    unsup_loss_sum += loss_unsup.item()
                    
                    loss = loss_sup + unsup_weight * loss_unsup
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA
            if ema:
                ema.update(model)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'mask': f'{mask_ratio:.3f}'
            })
        
        scheduler.step()
        
        # Validate
        val_acc, val_loss = validate(model, val_loader, device, criterion)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {total_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {100.*val_acc:.2f}%")
        
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️ 早停！驗證損失已經 {patience} 個 epoch 沒有改善")
                break
        
        # Best model storage
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"  ✅ 儲存最佳模型 (Val Acc: {100.*val_acc:.2f}%)")
    
    print(f"\n訓練完成！最佳驗證準確率: {100.*best_val_acc:.2f}%")
    return model


# ============================================================
# 7. 驗證函數
# ============================================================

def validate(model, val_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return accuracy, avg_loss

def evaluate_model(model, val_loader, device):
    print("\n" + "="*60)
    print("模型評估")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="評估中"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 分類報告
    print("\n分類報告:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Real', 'AI-Generated']))
    
    # 混淆矩陣
    print("\n混淆矩陣:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"                 預測Real  預測AI-Gen")
    print(f"實際Real          {cm[0][0]:>5}     {cm[0][1]:>5}")
    print(f"實際AI-Gen        {cm[1][0]:>5}     {cm[1][1]:>5}")



def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_files, train_labels, val_files, val_labels, unlabeled_files = prepare_dataset(
        real_dir='/kaggle/input/nycu-114-1-introduction-to-machine-learning-hw-4/data/train/real',           
        generated_dir='/kaggle/input/nycu-114-1-introduction-to-machine-learning-hw-4/data/train/generated',
        unlabeled_dir='/kaggle/input/nycu-114-1-introduction-to-machine-learning-hw-4/data/unlabeled',
        val_split=0.2,
        seed=42
    )
    
    train_transform = get_transforms(image_size=224, augment_type='weak')
    val_transform = get_transforms(image_size=224, augment_type='test')
    
    train_dataset = ImageDataset(train_files, train_labels, train_transform, USE_ALBUMENTATIONS)
    val_dataset = ImageDataset(val_files, val_labels, val_transform, USE_ALBUMENTATIONS)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    unlabeled_loader = None
    if len(unlabeled_files) > 0:
        unlabeled_dataset = ImageDataset(unlabeled_files, None, train_transform, USE_ALBUMENTATIONS)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    
    print("\n建立模型: convnext_base.fb_in1k")
    model = timm.create_model('convnext_base.fb_in1k', pretrained=True, num_classes=2, drop_rate=0.15)  # 加入 Dropout
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        unlabeled_loader=unlabeled_loader,
        epochs=10,
        lr=3e-5,                       
        device=device,
        use_semi_supervised=True,
        threshold=0.93,                 
        unsup_weight=0.65              
    )
    
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    evaluate_model(model, val_loader, device)
    
    print("\n✅ 完成！模型已儲存為 best_model.pth")


if __name__ == '__main__':
    main()