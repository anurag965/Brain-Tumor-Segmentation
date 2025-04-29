import os
import yaml # type: ignore
import math
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image
import numpy as np
import cv2 # type: ignore
from tqdm import tqdm # type: ignore
import albumentations as A # type: ignore
from albumentations.pytorch import ToTensorV2 # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, jaccard_score # type: ignore
from functools import partial
import torch.utils.checkpoint as checkpoint # type: ignore
import warnings
import random
from torch.optim.lr_scheduler import OneCycleLR
warnings.filterwarnings("ignore")

# Add at the top of the file after imports
torch.backends.cuda.max_memory_reserved = True
torch.backends.cudnn.benchmark = True

# Add these environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Check CUDA setup
def check_cuda():
    """Check and print CUDA information once"""
    print("\n=== CUDA Setup ===")
    print(f"PyTorch version: {torch.__version__}")  # Fixed version attribute
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    print("=================\n")

# Call check_cuda once
check_cuda()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""# *BEiT Backbone Implementation*"""

# Add this before creating the datasets
import glob

# Check the actual directory structure
def check_directory_structure(base_dir):
    print(f"\n--- Directory Structure Check for {base_dir} ---")

    # Check main directories
    train_img_dir = os.path.join(base_dir, "train", "images")
    train_mask_dir = os.path.join(base_dir, "train", "labels")
    val_img_dir = os.path.join(base_dir, "valid", "images")
    val_mask_dir = os.path.join(base_dir, "valid", "labels")

    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        exists = os.path.exists(dir_path)
        if exists:
            files = glob.glob(os.path.join(dir_path, "*"))
            file_count = len(files)
            print(f"Directory {dir_path}: {'Exists' if exists else 'MISSING'}, Contains {file_count} files")
            if file_count > 0:
                # Show some example files
                print(f"  Example files: {[os.path.basename(f) for f in files[:3]]}")
        else:
            print(f"Directory {dir_path}: MISSING")

    # Try to find the first image and its corresponding mask
    if os.path.exists(train_img_dir) and os.path.exists(train_mask_dir):
        img_files = glob.glob(os.path.join(train_img_dir, "*.jpg")) + \
                    glob.glob(os.path.join(train_img_dir, "*.png"))

        if img_files:
            # Get first image
            first_img = img_files[0]
            img_basename = os.path.splitext(os.path.basename(first_img))[0]

            # Look for matching mask
            potential_masks = [
                os.path.join(train_mask_dir, f"{img_basename}.txt"),
                os.path.join(train_mask_dir, f"{img_basename}.png"),
                os.path.join(train_mask_dir, f"{img_basename}.jpg")
            ]

            found_mask = None
            for mask_path in potential_masks:
                if os.path.exists(mask_path):
                    found_mask = mask_path
                    break

            print(f"\nMatching check:")
            print(f"  Image: {first_img}")
            print(f"  Mask: {found_mask if found_mask else 'NO MATCHING MASK'}")

            if found_mask and found_mask.endswith('.txt'):
                # Show the content of the txt mask
                with open(found_mask, 'r') as f:
                    mask_content = f.read()
                print(f"\nMask content (first 200 chars):\n{mask_content[:200]}")

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated within the interval [a, b].
        # (1) Generate values from the uniform distribution U(low, high).
        low = norm_cdf((a - mean) / std)
        high = norm_cdf((b - mean) / std)
        tensor.uniform_(low, high)
        # (2) Use the inverse CDF transform for the normal distribution.
        tensor.erfinv_()
        # (3) Transform to the desired mean and std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        # (4) Clamp to ensure values are still in the interval [a, b]
        tensor.clamp_(min=a, max=b)
        return tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Fix for PatchEmbed
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        self.patch_shape = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

# Fix for Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Fix for Block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Fix for BEiT
class BEiT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, use_abs_pos_emb=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

"""# *Segmentation Model*"""

# Define SegmentationHead
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Define the BEiT-based Semantic Segmentation model
class BEiTSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(BEiTSegmentation, self).__init__()
        
        # Initialize backbone with fixed parameters
        self.backbone = BEiT(
            img_size=256,
            patch_size=16,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=768,
            depth=6,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            use_abs_pos_emb=True
        )
        
        # Feature processing blocks with corrected dimensions
        self.feature_blocks = nn.ModuleList([
            # First block processes backbone output (768 channels)
            nn.Sequential(
                nn.Conv2d(768, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            # Subsequent blocks maintain consistent dimensions
            nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])

        # Decoder blocks with matching dimensions
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            )
        ])

        # Attention blocks with matching dimensions
        self.attention_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 1),
                nn.Sigmoid()
            )
        ])

        # Final convolution
        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Get features from backbone
        B = x.shape[0]
        features = self.backbone(x)
        
        # Remove CLS token and reshape to spatial format
        features = features[:, 1:, :]
        H = W = int(math.sqrt(features.shape[1]))
        features = features.transpose(1, 2).reshape(B, -1, H, W)
        
        # Process features and store intermediate results
        processed_features = []
        x = features
        
        # Apply feature blocks sequentially
        for block in self.feature_blocks:
            x = block(x)
            processed_features.append(x)
        
        # Decoder with skip connections
        x = processed_features[0]
        
        for i, (decoder, attention) in enumerate(zip(self.decoder_blocks, self.attention_blocks)):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            
            # Get skip connection
            skip = processed_features[i + 1]
            
            # Apply decoder
            x = decoder(x)
            
            # Match spatial dimensions if needed
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            # Apply skip connection and attention
            x = x + skip
            x = x * attention(x)
        
        # Final processing
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        
        return x

"""# *Dataset and Training Functions*"""

#---------------  ---------------#

# Load dataset configuration
def load_data_config(yaml_path):
    """Load dataset configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Custom Dataset for semantic segmentation
class TumorSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, num_classes=3):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes

        # Support both extensions
        self.image_files = [f for f in os.listdir(image_dir)
                          if f.endswith(('.jpg', '.png', '.jpeg'))]

        # For each image, find the corresponding mask
        self.valid_pairs = []
        for img_file in self.image_files:
            img_name = os.path.splitext(img_file)[0]
            mask_file = None

            # Look for masks with different extensions
            for ext in ['.txt', '.png', '.jpg']:
                candidate = os.path.join(mask_dir, img_name + ext)
                if os.path.exists(candidate):
                    mask_file = img_name + ext
                    break

            if mask_file:
                self.valid_pairs.append((img_file, mask_file))

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.valid_pairs[idx]

        # Load image
        img_path = os.path.join(self.image_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Handle different mask formats
        if mask_file.endswith('.txt'):  # YOLO format
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.int64)

            try:
                with open(mask_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id >= self.num_classes:
                            print(f"Warning: Invalid class ID {class_id} in {mask_path}")
                            continue

                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                        x1 = max(0, int((x_center - bbox_width/2) * w))
                        y1 = max(0, int((y_center - bbox_height/2) * h))
                        x2 = min(w-1, int((x_center + bbox_width/2) * w))
                        y2 = min(h-1, int((y_center + bbox_height/2) * h))

                        mask[y1:y2+1, x1:x2+1] = class_id
            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")
                mask = np.zeros((h, w), dtype=np.int64)
        else:  # Image format
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.int64)
            mask[mask >= self.num_classes] = 0

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()

        return image, mask

# Modified training loop
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            
            if pbar.n % 10 == 0:
                torch.cuda.empty_cache()
                
    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, criterion, device, num_classes, epoch=None):
    model.eval()
    total_loss = 0
    
    # Initialize confusion matrix
    confusion_mat = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validating") as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                
                # Update confusion matrix
                for t, p in zip(targets.flatten(), preds.flatten()):
                    confusion_mat[t, p] += 1
                
                pbar.set_postfix(loss=loss.item())
    
    # Plot confusion matrix if epoch is provided
    if epoch is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='g', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_epoch_{epoch+1}.png')
        plt.close()
    
    # Calculate metrics from confusion matrix
    metrics = {}
    class_ious = []
    
    for cls in range(num_classes):
        tp = confusion_mat[cls, cls]
        fp = confusion_mat[:, cls].sum() - tp
        fn = confusion_mat[cls, :].sum() - tp
        tn = confusion_mat.sum() - (tp + fp + fn)
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        class_ious.append(iou)
        
        metrics[f'class_{cls}'] = {
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1-Score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'IoU': iou
        }
    
    # Calculate mean IoU
    mean_iou = np.mean(class_ious)
    
    # Print validation metrics
    print("\nValidation Metrics:")
    print("-" * 50)
    print(f"Mean IoU: {mean_iou:.4f}")
    for cls in range(num_classes):
        print(f"\nClass {cls} metrics:")
        for metric, value in metrics[f'class_{cls}'].items():
            print(f"{metric}: {value:.4f}")
    
    return total_loss / len(dataloader), mean_iou, class_ious

# Visualization function
def visualize_prediction(model, test_img_path, device, class_names, save_path=None):
    """Visualize prediction on a single test image with improved overlay"""
    # Prepare image
    image = cv2.imread(test_img_path)
    if image is None:
        print(f"Error loading image: {test_img_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Store original dimensions
    orig_h, orig_w = image.shape[:2]

    # Resize image to model input size
    target_size = (256, 256)
    resized_image = cv2.resize(image, target_size)

    # Apply transforms
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Transform and add batch dimension
    transformed = transform(image=resized_image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()

    # Resize prediction mask back to original image size
    pred_mask = cv2.resize(pred_mask.astype(float), (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST).astype(int)

    # Create figure
    plt.figure(figsize=(15, 5))

    # 1. Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # 2. Prediction mask
    plt.subplot(1, 3, 2)
    # Create a colored mask
    mask_display = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    tumor_regions = (pred_mask == 1)
    mask_display[tumor_regions] = [255, 0, 0]  # Red for tumor
    plt.imshow(mask_display)
    plt.title('Predicted Segmentation')
    plt.axis('off')

    # 3. Overlay
    plt.subplot(1, 3, 3)
    overlay = image.copy()

    # Only create overlay if tumor regions are detected
    if np.any(tumor_regions):
        # Create tumor overlay
        mask_overlay = np.zeros_like(image)
        mask_overlay[tumor_regions] = [255, 0, 0]

        # Blend images
        alpha = 0.5
        overlay = cv2.addWeighted(image, 1, mask_overlay, alpha, 0)

        # Add contours
        tumor_mask_uint8 = tumor_regions.astype(np.uint8)
        contours, _ = cv2.findContours(tumor_mask_uint8,
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    plt.imshow(overlay)
    plt.title('Overlay with Tumor Regions' if np.any(tumor_regions) else 'No Tumor Detected')
    plt.axis('off')

    # Save with high quality
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()

    # Print detection status
    if not np.any(tumor_regions):
        print(f"No tumor regions detected in {os.path.basename(test_img_path)}")

"""# *Loading Dataset*"""

# Load config
config = load_data_config("brain tumor.v2-release.yolov7pytorch/data.yaml")
num_classes = config['nc']
class_names = config['names']
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Setup paths relative to yaml location
base_dir = os.path.dirname("brain tumor.v2-release.yolov7pytorch/data.yaml")
train_img_dir = os.path.join(base_dir, "train/images")
train_mask_dir = os.path.join(base_dir, "train/labels")
val_img_dir = os.path.join(base_dir, "valid/images")
val_mask_dir = os.path.join(base_dir, "valid/labels")

print(f"Using device: {device}")
check_directory_structure("brain tumor.v2-release.yolov7pytorch")

# Transforms with smaller image size
train_transform = A.Compose([
    A.Resize(256, 256),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        A.RandomGamma(gamma_limit=(80, 120), p=1),
    ], p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
        A.GridDistortion(p=1),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
    ], p=0.3),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),  # Reduced from 384
    A.Normalize(),
    ToTensorV2()
])

# Datasets and loaders
train_dataset = TumorSegmentationDataset(
    train_img_dir,
    train_mask_dir,
    transform=train_transform,
    num_classes=num_classes  # Now using 3 classes from yaml
)

val_dataset = TumorSegmentationDataset(
    val_img_dir,
    val_mask_dir,
    transform=val_transform,
    num_classes=num_classes
)

# Add validation before training
print("\nValidating class indices...")
max_train = max(mask.max() for _, mask in train_dataset)
max_val = max(mask.max() for _, mask in val_dataset)
print(f"Max class index in train: {max_train}")
print(f"Max class index in val: {max_val}")
print(f"Number of classes: {num_classes}")

assert max_train < num_classes, f"Invalid class index in train set: {max_train} >= {num_classes}"
assert max_val < num_classes, f"Invalid class index in val set: {max_val} >= {num_classes}"

# Validate class indices
print("\nValidating mask values...")
for dataset, name in [(train_dataset, 'train'), (val_dataset, 'val')]:
    unique_classes = set()
    for _, mask in dataset:
        unique_classes.update(mask.unique().tolist())
    print(f"{name} dataset unique classes: {sorted(unique_classes)}")
    assert max(unique_classes) < num_classes, \
        f"Invalid class index in {name} set: {max(unique_classes)} >= {num_classes}"

print("\nDataset Statistics:")
print("-" * 50)
print(f"Images in train directory: {len(os.listdir(train_img_dir))}")
print(f"Masks in train directory: {len(os.listdir(train_mask_dir))}")
print("\nSample image-mask pairs:")
print("-" * 50)
print("First 5 train pairs:", train_dataset.valid_pairs[:5])
print(f"\nTotal valid pairs:")
print("-" * 50)
print(f"Train dataset: {len(train_dataset)} pairs")
print(f"Val dataset: {len(val_dataset)} pairs")

# Add before training loop
def calculate_class_weights(dataset):
    class_counts = torch.zeros(num_classes)
    for _, mask in dataset:
        for i in range(num_classes):
            class_counts[i] += (mask == i).sum()
    
    # Calculate weights inversely proportional to class frequencies
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    return weights.to(device)

class_weights = calculate_class_weights(train_dataset)

# Define DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.smooth = 1
        self.weight = weight

    def forward(self, inputs, targets):
        # Convert inputs to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Flatten the tensors
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)
        
        # Calculate Dice coefficient for each class
        intersection = (inputs * targets).sum(dim=2)
        union = inputs.sum(dim=2) + targets.sum(dim=2)
        
        # Calculate Dice loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Apply weights if provided
        if self.weight is not None:
            dice = dice * self.weight
            
        # Average over classes and batches
        return 1 - dice.mean()

# Combine CE and Dice loss
class CombinedLoss(nn.Module):
    def __init__(self, weight=None):  # Changed from weights to weight
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(weight=weight)
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return ce_loss + dice_loss

# Update criterion initialization
criterion = CombinedLoss(weight=class_weights)  # Changed from weights to weight

# Add this before model, optimizer and scheduler setup
batch_size = 2  # Reduced batch size
num_epochs = 50

# Create data loaders first
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,  # Changed to False for validation
    num_workers=0,
    pin_memory=True
)

# Initialize model first
model = BEiTSegmentation(num_classes=num_classes).to(device)

# Then initialize optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    div_factor=10,
    final_div_factor=100
)

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device):
    """Training function with validation"""
    best_miou = 0
    best_model_state = None
    
    print("Starting Training Phase...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training
        model.train()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validation
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            val_loss, val_miou, _ = validate(model, val_loader, criterion, device, num_classes, epoch)
            print(f"Validation Loss: {val_loss:.4f}, Mean IoU: {val_miou:.4f}")
            
            if val_miou > best_miou:
                best_miou = val_miou
                best_model_state = model.state_dict().copy()
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_miou,
                    'val_loss': val_loss
                }, "best_beit_segmentation.pth")
                print("New best model saved!")
        
        torch.cuda.empty_cache()
    
    return best_model_state

def test_model(model, test_image_paths, device, class_names):
    """Testing function"""
    print("\nStarting Testing Phase...")
    model.eval()
    results = []
    
    for img_path in tqdm(test_image_paths, desc="Testing"):
        # Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        
        transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Get prediction
        with torch.no_grad():
            input_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
            predicted_class = int(np.bincount(prediction.flatten()).argmax())
            confidence = float(probabilities[0, predicted_class].mean().cpu())
        
        # Save visualization
        save_path = f'test_predictions/{os.path.basename(img_path)}'
        os.makedirs('test_predictions', exist_ok=True)
        visualize_prediction(model, img_path, device, class_names, save_path)
        
        results.append({
            'image': os.path.basename(img_path),
            'prediction': predicted_class,
            'confidence': confidence
        })
    
    return results

if __name__ == "__main__":
    # Setup
    model = BEiTSegmentation(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-2,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )
    
    # Training Phase
    best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # Load best model for testing
    model.load_state_dict(best_model_state)
    
    # Testing Phase
    test_img_dir = os.path.join(base_dir, "test/images")
    test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if test_images:
        print(f"\nFound {len(test_images)} test images")
        test_results = test_model(model, test_images, device, class_names)
        
        print("\nTest Results Summary:")
        print("-" * 50)
        for result in test_results:
            class_name = class_names[result['prediction']] if result['prediction'] < len(class_names) else 'Unknown'
            print(f"Image: {result['image']}")
            print(f"Predicted Class: {class_name}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("-" * 30)
    else:
        print("No test images found!")

    print("\nComplete pipeline finished!")
