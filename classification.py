import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import os
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F

# CBAM module (simplified)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        return x

# VGG19 + CBAM
class VGG19_CBAM(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19_CBAM, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = vgg19.features
        self.cbam = CBAM(512)  # 512 is the last feature map channels in VGG19
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Grad-CAM utility
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        weights = gradients.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activations).sum(0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        return cam

# Custom dataset for binary masks in class folders
class MaskFolderDataset(torch.utils.data.Dataset):
    def __init__(self, mask_root, class_names, transform=None):
        self.mask_root = mask_root
        self.class_names = class_names  # e.g., ['l1', 'l2', 'l3', 'n']
        self.transform = transform
        self.samples = []
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(mask_root, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        mask_path, label = self.samples[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            mask = cv2.resize(mask, (224, 224))
            mask_img = np.stack([mask]*3, axis=-1)  # fake 3-channel image
            mask_img = self.transform(mask_img)
        else:
            mask_img = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(cv2.resize(mask, (224, 224))).float() / 255.0
        return mask_img, mask_tensor, label

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def show_gradcam_on_image(img, mask, cam, save_path, mask_save_path):
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(img)
    overlay = overlay / np.max(overlay)
    plt.imsave(save_path, np.uint8(255 * overlay))
    # Save the binary mask as well
    plt.imsave(mask_save_path, np.uint8(255 * mask.cpu().numpy()))

def generate_gradcam_images(model, dataloader, device, output_dir='./gradcam_results'):
    os.makedirs(output_dir, exist_ok=True)
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    model.eval()
    for i, (images, masks, labels) in enumerate(dataloader):
        images = images.to(device)
        for j in range(min(4, images.size(0))):
            input_img = images[j:j+1]
            cam = gradcam(input_img, class_idx=labels[j].item())
            save_path = f"{output_dir}/gradcam_{i}_{j}.png"
            mask_save_path = f"{output_dir}/mask_{i}_{j}.png"
            show_gradcam_on_image(input_img[0].cpu(), masks[j], cam, save_path, mask_save_path)
        break
    gradcam.remove_hooks()

def get_class_weights(dataset, num_classes):
    # Count samples per class
    counts = [0] * num_classes
    for _, _, label in dataset:
        counts[label] += 1
    # Inverse frequency for each class
    weights = [0.] * num_classes
    for i in range(num_classes):
        weights[i] = 1.0 / counts[i] if counts[i] > 0 else 0
    # Assign weight to each sample
    sample_weights = [weights[label] for _, _, label in dataset]
    return sample_weights, weights

# Training setup for image+mask+class dataset
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_folder = r'classification_dataset'  # <-- Update this path
    class_names = ['Brown_spots', 'healthy', 'white_scale']        # <-- Update class names
    num_classes = len(class_names)
    trainset = MaskFolderDataset(mask_folder, class_names, transform=get_transforms(train=True))
    valset = MaskFolderDataset(mask_folder, class_names, transform=get_transforms(train=False))
    total_len = len(trainset)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len
    train_subset, val_subset = random_split(trainset, [train_len, val_len])

    # Compute sample weights for the training subset
    train_indices = train_subset.indices if hasattr(train_subset, 'indices') else train_subset
    train_labels = [trainset[i][2] for i in train_indices]
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    # Assign weight to each sample
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    trainloader = DataLoader(train_subset, batch_size=8, sampler=sampler)
    valloader = DataLoader(val_subset, batch_size=8, shuffle=False)
    model = VGG19_CBAM(num_classes=num_classes).to(device)
    # Use class weights in loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Ensure all plot directories exist before saving
    os.makedirs('./plots/loss_acc', exist_ok=True)
    os.makedirs('./plots/confmat', exist_ok=True)
    os.makedirs('./gradcam_results', exist_ok=True)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)

    # Recommended: 30-50 epochs with early stopping
    # For your dataset size and early stopping, 30-50 is usually sufficient.
    for epoch in range(50):  # up to 50 epochs, will stop early if needed
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, masks, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, masks, labels in valloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_loss_epoch = val_loss / val_total
        val_acc_epoch = val_correct / val_total
        val_losses.append(val_loss_epoch)
        val_accuracies.append(val_acc_epoch)

        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss_epoch:.4f} Acc: {val_acc_epoch:.4f}")

        # Early stopping check
        early_stopping(val_loss_epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), "vgg19_cbam_imageonly.pth")
    print("Model weights saved to vgg19_cbam_imageonly.pth")

    # Plot and save loss and accuracy
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.savefig('./plots/loss_acc/train_vs_val_accuracy.png')
    plt.close()

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.savefig('./plots/loss_acc/train_vs_val_loss.png')
    plt.close()

    # Evaluate and plot confusion matrix and classification report on validation set
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_features = []
    with torch.no_grad():
        for images, masks, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            # For t-SNE/PCA, use features before classifier
            feats = model.features(images)
            feats = model.cbam(feats)
            feats = model.avgpool(feats)
            feats = torch.flatten(feats, 1)
            all_features.append(feats.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('./plots/confmat/confmat_val.png')
    plt.close()

    # Classification Report (Precision, Recall, F1-score)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ['precision', 'recall', 'f1-score']
    for i, metric in enumerate(metrics):
        vals = [report[c][metric] for c in class_names]
        ax.bar(np.arange(len(class_names)) + i*0.25, vals, width=0.25, label=metric)
    ax.set_xticks(np.arange(len(class_names)) + 0.25)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, F1-score per Class')
    ax.legend()
    plt.tight_layout()
    plt.savefig('./plots/confmat/precision_recall_f1.png')
    plt.close()

    # ROC-AUC Curve (macro/micro)
    y_true = np.array(all_labels)
    y_score = np.array(all_probs)
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Macro-average ROC
    fpr["macro"], tpr["macro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], label=f'macro-average ROC (area = {roc_auc["macro"]:.2f})', color='navy', linestyle=':')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")
    plt.savefig('./plots/confmat/roc_auc.png')
    plt.close()

    # t-SNE plot (feature space visualization)
    try:
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(all_features)
        plt.figure(figsize=(6, 6))
        for idx, cname in enumerate(class_names):
            plt.scatter(features_2d[y_true == idx, 0], features_2d[y_true == idx, 1], label=cname, alpha=0.6)
        plt.legend()
        plt.title('t-SNE Feature Space')
        plt.savefig('./plots/confmat/tsne_features.png')
        plt.close()
    except Exception as e:
        print(f"t-SNE plot failed: {e}")

    # PCA plot (feature space visualization)
    try:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)
        plt.figure(figsize=(6, 6))
        for idx, cname in enumerate(class_names):
            plt.scatter(features_2d[y_true == idx, 0], features_2d[y_true == idx, 1], label=cname, alpha=0.6)
        plt.legend()
        plt.title('PCA Feature Space')
        plt.savefig('./plots/confmat/pca_features.png')
        plt.close()
    except Exception as e:
        print(f"PCA plot failed: {e}")

    generate_gradcam_images(model, trainloader, device)

if __name__ == "__main__":
    print("All requirements are complete and class balancing is set. Ready to start training and evaluation.")
    train()