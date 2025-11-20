import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import os

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

class TBXrayClassifier(nn.Module):
    """
    Flexible TB X-ray classifier supporting multiple architectures
    Compatible with pytorch-grad-cam
    """

    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=True):
        """
        Args:
            model_name: One of ['vgg16', 'vgg19', 'resnet50', 'resnet101',
                        'efficientnet_b0', 'efficientnet_b1', 'densenet121',
                        'densenet169', 'mobilenet_v3_large', 'mobilenet_v3_small']
            num_classes: Number of output classes (default: 2 for TB/Normal)
            pretrained: Use pretrained weights
        """
        super(TBXrayClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load the appropriate model
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'features'

        elif model_name == 'vgg19':
            self.model = models.vgg19(pretrained=pretrained)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'features'

        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'layer4'

        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'layer4'

        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'features'

        elif model_name == 'efficientnet_b1':
            self.model = models.efficientnet_b1(pretrained=pretrained)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'features'

        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'features'

        elif model_name == 'densenet169':
            self.model = models.densenet169(pretrained=pretrained)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'features'

        elif model_name == 'mobilenet_v3_large':
            self.model = models.mobilenet_v3_large(pretrained=pretrained)
            num_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'features'

        elif model_name == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(pretrained=pretrained)
            num_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_features, num_classes)
            self.target_layer_name = 'features'

        else:
            raise ValueError(f"Model {model_name} not supported. Choose from: "
                             "vgg16, vgg19, resnet50, resnet101, efficientnet_b0, "
                             "efficientnet_b1, densenet121, densenet169, "
                             "mobilenet_v3_large, mobilenet_v3_small")

    def forward(self, x):
        return self.model(x)

    def get_target_layer(self):
        """Returns the target layer for Grad-CAM visualization"""
        if self.target_layer_name == 'features':
            return getattr(self.model, self.target_layer_name)[-1]
        elif self.target_layer_name == 'layer4':
            return getattr(self.model, self.target_layer_name)[-1]
        else:
            raise ValueError(f"Unknown target layer: {self.target_layer_name}")

    def freeze_backbone(self, freeze_until='all'):
        """
        Freeze layers for transfer learning
        Args:
            freeze_until: 'all' (freeze all backbone), 'half' (freeze first half),
                         'none' (freeze nothing)
        """
        if freeze_until == 'none':
            return

        if self.model_name in ['vgg16', 'vgg19']:
            layers = self.model.features
        elif self.model_name in ['resnet50', 'resnet101']:
            layers = [self.model.conv1, self.model.bn1, self.model.layer1,
                      self.model.layer2, self.model.layer3]
        elif 'efficientnet' in self.model_name:
            layers = self.model.features
        elif 'densenet' in self.model_name:
            layers = self.model.features
        elif 'mobilenet' in self.model_name:
            layers = self.model.features

        if freeze_until == 'all':
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False
        elif freeze_until == 'half':
            freeze_count = len(list(layers)) // 2
            for i, layer in enumerate(layers):
                if i < freeze_count:
                    for param in layer.parameters():
                        param.requires_grad = False


def get_transforms(model_name='efficientnet_b0'):
    """Returns preprocessing transforms appropriate for the model"""
    if 'efficientnet' in model_name or 'mobilenet' in model_name:
        # These models may have slightly different preprocessing
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # Standard ImageNet preprocessing
        return transforms.Compose([
            transforms.Resize((224, 224)),
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
def generate_gradcam(model, img_path, target_class=None):
    """
    Generate Grad-CAM visualization for an X-ray image

    Args:
        model: Trained TBXrayClassifier model
        img_path: Path to the X-ray image
        target_class: Target class for Grad-CAM (None for predicted class)

    Returns:
        cam_image: Grad-CAM visualization overlaid on original image
        prediction: Model prediction class
        probabilities: Class probabilities
    """
    model.eval()

    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = get_transforms(model.model_name)(img).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
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

    return cam_image, pred_class, pred[0].numpy()


def compare_models(model_names=['efficientnet_b0', 'resnet50', 'densenet121']):
    """Compare different model architectures"""
    print("Model Comparison")
    print("=" * 80)
    print(f"{'Model':<20} {'Parameters':<15} {'Trainable':<15} {'Size (MB)':<12}")
    print("-" * 80)

    for name in model_names:
        model = TBXrayClassifier(model_name=name, pretrained=True)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32

        print(f"{name:<20} {total_params:>13,} {trainable_params:>13,} {size_mb:>10.2f}")
    print("=" * 80)


# Example usage
if __name__ == "__main__":
    # Compare different models
    compare_models(['vgg16', 'resnet50', 'efficientnet_b0', 'densenet121', 'mobilenet_v3_large'])

    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)

    # Example 1: Create EfficientNet model (recommended)
    print("\n1. Create EfficientNet-B0 model (Recommended):")
    model = TBXrayClassifier(model_name='efficientnet_b0', num_classes=2, pretrained=True)
    print(f"   Created {model.model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Example 2: Create ResNet model
    print("\n2. Create ResNet50 model:")
    model_resnet = TBXrayClassifier(model_name='resnet50', num_classes=2, pretrained=True)
    print(f"   Created {model_resnet.model_name} with {sum(p.numel() for p in model_resnet.parameters()):,} parameters")

    # Example 3: Freeze backbone for transfer learning
    print("\n3. Freeze backbone for transfer learning:")
    model.freeze_backbone(freeze_until='all')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters after freezing: {trainable:,}")

    # Example 4: Test forward pass
    print("\n4. Test forward pass:")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Probabilities: {torch.softmax(output, dim=1).numpy()}")

    normal_dir = '../data/TB_Chest_Radiography_Database/Normal/'  # Directory containing normal X-rays
    tb_dir = '../data/TB_Chest_Radiography_Database/Tuberculosis/'  # Directory containing TB X-rays

# Choose your model architecture
    model_name = 'efficientnet_b0'  # or 'resnet50', 'densenet121', etc.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = TBXrayClassifier(model_name=model_name, num_classes=2, pretrained=True)
    model = model.to(device)

    # Optional: Freeze backbone for faster training
    model.freeze_backbone(freeze_until='all')  # or 'half', 'none'

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.1, patience=5)

    train_loader, val_loader, class_dist, class_weights = create_dataloaders(
        normal_dir=normal_dir,
        tb_dir=tb_dir,
        batch_size=32,
        val_split=0.2,
        num_workers=4
    )
    num_epochs = 10
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {accuracy:.2f}%')

    # Generate Grad-CAM
    cam_img, pred_class, probs = generate_gradcam(model,
                                                  '../data/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-1.png')
    print(f"\\nPrediction: {'TB' if pred_class == 1 else 'Normal'}")
    print(f"Confidence: {probs[pred_class] * 100:.2f}%")
    print(f"Probabilities - Normal: {probs[0] * 100:.2f}%, TB: {probs[1] * 100:.2f}%")
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
