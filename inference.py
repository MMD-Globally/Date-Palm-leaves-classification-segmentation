import torch
from torchvision import transforms
from PIL import Image
import os
from classification import VGG19_CBAM, get_transforms

# Define your class names
class_names = ['Brown_spots', 'healthy', 'white_scale']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG19_CBAM(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(r'D:\New Date Palm data\vgg19_cbam_imageonly.pth', map_location=device))
model.eval()

# Define the same transforms as used during validation
transform = get_transforms(train=False)

def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        return class_names[class_idx]

# Example usage:
test_image_path = r'D:\New Date Palm data\classification_dataset\Brown_spots\brownspots-1.jpg'  # Change to your test image
predicted_class = classify_image(test_image_path)
print(f"Predicted class: {predicted_class}")