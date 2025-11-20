import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class TBXrayDataset(Dataset):
    """
    Custom Dataset for TB X-ray images from separate directories
    """

    def __init__(self, normal_dir, tb_dir, transform=None, test_size=0.0, random_state=42, is_test=False):
        """
        Args:
            normal_dir: Path to directory containing normal X-ray images
            tb_dir: Path to directory containing TB X-ray images
            transform: Optional transform to be applied on images
            test_size: Proportion of data to use for testing (0.0 to 1.0)
            random_state: Random seed for reproducibility
            is_test: If True, load test split; if False, load train split
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load normal X-rays (label = 0)
        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(normal_dir, img_name))
                    self.labels.append(0)
        else:
            print(f"Warning: Normal directory not found: {normal_dir}")

        # Load TB X-rays (label = 1)
        if os.path.exists(tb_dir):
            for img_name in os.listdir(tb_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(tb_dir, img_name))
                    self.labels.append(1)
        else:
            print(f"Warning: TB directory not found: {tb_dir}")

        # Split into train/test if needed
        if test_size > 0.0 and len(self.image_paths) > 0:
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                self.image_paths, self.labels,
                test_size=test_size,
                random_state=random_state,
                stratify=self.labels
            )

            if is_test:
                self.image_paths = test_paths
                self.labels = test_labels
            else:
                self.image_paths = train_paths
                self.labels = train_labels

        print(f"Loaded {len(self.image_paths)} images: "
              f"{self.labels.count(0)} Normal, {self.labels.count(1)} TB")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """Returns the number of samples per class"""
        normal_count = self.labels.count(0)
        tb_count = self.labels.count(1)
        return {'Normal': normal_count, 'TB': tb_count}


class TBDenseNetClassifier(nn.Module):
    """
    DenseNet121-based TB X-ray classifier compatible with pytorch-grad-cam

    DenseNet is particularly well-suited for medical imaging because:
    - Dense connections promote feature reuse and gradient flow
    - Requires fewer parameters than VGG while being more accurate
    - Used in CheXNet, a state-of-the-art chest X-ray classifier
    """

    def __init__(self, num_classes=2, pretrained=True, densenet_version='121'):
        """
        Args:
            num_classes: Number of output classes (default: 2 for TB/Normal)
            pretrained: Use pretrained ImageNet weights
            densenet_version: '121', '161', '169', or '201'
        """
        super(TBDenseNetClassifier, self).__init__()

        self.densenet_version = densenet_version

        # Load pretrained DenseNet
        if densenet_version == '121':
            self.densenet = models.densenet121(pretrained=pretrained)
        elif densenet_version == '161':
            self.densenet = models.densenet161(pretrained=pretrained)
        elif densenet_version == '169':
            self.densenet = models.densenet169(pretrained=pretrained)
        elif densenet_version == '201':
            self.densenet = models.densenet201(pretrained=pretrained)
        else:
            raise ValueError(f"DenseNet version {densenet_version} not supported. "
                             "Choose from: 121, 161, 169, 201")

        # Get number of input features to the classifier
        num_features = self.densenet.classifier.in_features

        # Replace classifier with custom layer for TB classification
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet(x)

    def get_target_layer(self):
        """
        Returns the target layer for Grad-CAM visualization
        For DenseNet, we use the last convolutional layer in the last dense block
        """
        # For DenseNet121: features.denseblock4 is the last dense block
        # We target the last denselayer's conv2 (the last conv layer)
        if self.densenet_version == '121':
            return self.densenet.features.denseblock4.denselayer16.conv2
        elif self.densenet_version == '161':
            return self.densenet.features.denseblock4.denselayer24.conv2
        elif self.densenet_version == '169':
            return self.densenet.features.denseblock4.denselayer32.conv2
        elif self.densenet_version == '201':
            return self.densenet.features.denseblock4.denselayer32.conv2
        else:
            # Fallback to transition layer before final norm
            return self.densenet.features.denseblock4

    def freeze_features(self, freeze_until='all'):
        """
        Freeze feature extraction layers for transfer learning

        Args:
            freeze_until: 'all' - freeze all feature layers
                         'half' - freeze first two dense blocks
                         'none' - freeze nothing
        """
        if freeze_until == 'none':
            return

        if freeze_until == 'all':
            # Freeze all feature layers
            for param in self.densenet.features.parameters():
                param.requires_grad = False

        elif freeze_until == 'half':
            # Freeze first two dense blocks (denseblock1 and denseblock2)
            for name, param in self.densenet.features.named_parameters():
                if 'denseblock1' in name or 'denseblock2' in name:
                    param.requires_grad = False

    def get_model_info(self):
        """Returns information about the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'architecture': f'DenseNet-{self.densenet_version}',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }


def get_transforms(augment=False):
    """
    Returns preprocessing transforms for DenseNet

    Args:
        augment: If True, includes data augmentation for training
    """
    if augment:
        # Training transforms with data augmentation
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(normal_dir, tb_dir, batch_size=32, val_split=0.2, num_workers=4):
    """
    Create train and validation dataloaders from separate directories
    Handles class imbalance automatically

    Args:
        normal_dir: Path to normal X-ray images directory
        tb_dir: Path to TB X-ray images directory
        batch_size: Batch size for training
        val_split: Proportion of data for validation (0.0 to 1.0)
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, class_distribution, class_weights
    """

    # Get transforms (augmented for training, standard for validation)
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    # Create datasets
    train_dataset = TBXrayDataset(
        normal_dir=normal_dir,
        tb_dir=tb_dir,
        transform=train_transform,
        test_size=val_split,
        is_test=False
    )

    val_dataset = TBXrayDataset(
        normal_dir=normal_dir,
        tb_dir=tb_dir,
        transform=val_transform,
        test_size=val_split,
        is_test=True
    )

    # Calculate class weights for handling imbalance
    train_dist = train_dataset.get_class_distribution()
    normal_count = train_dist['Normal']
    tb_count = train_dist['TB']
    total = normal_count + tb_count

    # Compute inverse frequency weights
    class_weights = torch.FloatTensor([
        total / (2 * normal_count) if normal_count > 0 else 1.0,  # Weight for Normal
        total / (2 * tb_count) if tb_count > 0 else 1.0  # Weight for TB
    ])

    print(f"\nClass Distribution:")
    print(f"  Training - Normal: {normal_count}, TB: {tb_count}")
    print(f"  Imbalance Ratio: {max(normal_count, tb_count) / min(normal_count, tb_count):.2f}:1")
    print(f"  Class Weights: Normal={class_weights[0]:.4f}, TB={class_weights[1]:.4f}")

    # Create weighted sampler for training (optional but recommended)
    train_labels = train_dataset.labels
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get class distribution
    val_dist = val_dataset.get_class_distribution()
    print(f"  Validation - Normal: {val_dist['Normal']}, TB: {val_dist['TB']}")

    return train_loader, val_loader, (train_dist, val_dist), class_weights


def generate_gradcam(model, img_path, target_class=None, device='cpu'):
    """
    Generate Grad-CAM visualization for an X-ray image

    Args:
        model: Trained TBDenseNetClassifier model
        img_path: Path to the X-ray image
        target_class: Target class for Grad-CAM (None for predicted class)
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        cam_image: Grad-CAM visualization overlaid on original image
        prediction: Model prediction class
        probabilities: Class probabilities
    """
    # Ensure model is on correct device and in eval mode
    model = model.to(device)
    model.eval()

    # Enable gradient computation for the model
    for param in model.parameters():
        param.requires_grad = True

    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = get_transforms(augment=False)(img).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # Get prediction
    output = model(img_tensor)
    pred = torch.softmax(output, dim=1)
    pred_class = torch.argmax(pred, dim=1).item()

    # Setup Grad-CAM
    target_layer = model.get_target_layer()
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Generate CAM
    targets = [target_class] if target_class is not None else None
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Overlay on original image
    img_np = np.array(img.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    return cam_image, pred_class, pred[0].detach().cpu().numpy()


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("DenseNet121 TB X-ray Classifier")
    print("=" * 80)

    # Initialize model
    model = TBDenseNetClassifier(num_classes=2, pretrained=True, densenet_version='121')

    # Display model information
    info = model.get_model_info()
    print(f"\nModel Architecture: {info['architecture']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Sample probabilities: {torch.softmax(output, dim=1).numpy()}")

    # Show Grad-CAM target layer
    print(f"\nGrad-CAM target layer: {model.get_target_layer()}")

    print("\n" + "=" * 80)
    print("COMPLETE TRAINING PIPELINE")
    print("=" * 80)

    normal_dir = '../data/TB_Chest_Radiography_Database/Normal/'  # Directory containing normal X-rays
    tb_dir = '../data/TB_Chest_Radiography_Database/Tuberculosis/'  # Directory containing TB X-rays

    # Create dataloaders with data augmentation
    train_loader, val_loader, class_dist, class_weights = create_dataloaders(
        normal_dir=normal_dir,
        tb_dir=tb_dir,
        batch_size=32,
        val_split=0.2,
        num_workers=4
    )

    # ============================================================================
    # STEP 2: Initialize DenseNet Model
    # ============================================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create DenseNet121 model (you can also use '161', '169', or '201')
    model = TBDenseNetClassifier(
        num_classes=2,
        pretrained=True,
        densenet_version='121'
    )
    model = model.to(device)

    # Optional: Freeze feature layers for transfer learning
    model.freeze_features(freeze_until='all')  # or 'half', 'none'

    # Display model info
    info = model.get_model_info()
    print(f"Training {info['architecture']}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")

    # ============================================================================
    # STEP 3: Training Setup
    # ============================================================================

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )

    # ============================================================================
    # STEP 4: Training Loop
    # ============================================================================

    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_densenet121_tb.pth')
            print(f'✓ Saved new best model with validation accuracy: {val_accuracy:.2f}%')

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    # ============================================================================
    # STEP 5: Load Best Model and Evaluate
    # ============================================================================

    # Load best model
    # model.load_state_dict(torch.load('best_densenet121_tb.pth'))
    # model.eval()

    # Generate Grad-CAM for a sample image
    print(model.get_target_layer())
    cam_img, pred_class, probs = generate_gradcam(model, '../data/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-1.png')
    print(f"\\nPrediction: {'TB' if pred_class == 1 else 'Normal'}")
    print(f"Confidence: {probs[pred_class]*100:.2f}%")
    print(f"Probabilities - Normal: {probs[0]*100:.2f}%, TB: {probs[1]*100:.2f}%")
    img = Image.fromarray(cam_img)
    img.save('./TB.jpeg')
    # plt.imshow(cam_img)

    cam_img, pred_class, probs = generate_gradcam(model,
                                                  '../data/TB_Chest_Radiography_Database/Normal/Normal-1.png')
    print(f"\\nPrediction: {'TB' if pred_class == 1 else 'Normal'}")
    print(f"Confidence: {probs[pred_class] * 100:.2f}%")
    print(f"Probabilities - Normal: {probs[0] * 100:.2f}%, TB: {probs[1] * 100:.2f}%")
    img = Image.fromarray(cam_img)
    img.save('./Normal.jpeg')
    # plt.imshow(cam_img)

    # ============================================================================
    # OPTIONAL: Fine-tune with Unfrozen Layers
    # ============================================================================

    # After initial training, you can unfreeze and fine-tune
    model.freeze_features(freeze_until='none')  # Unfreeze all layers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

    # Continue training for a few more epochs...


    print("\n" + "=" * 80)
    print("WHY DENSENET FOR TB X-RAYS?")
    print("=" * 80)
