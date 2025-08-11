import streamlit as st
from PIL import Image
import os
import torch
from torchvision import transforms
from classification import VGG19_CBAM
import requests
from streamlit_lottie import st_lottie
import json

# Set page config
st.set_page_config(
    page_title="Date Palm Disease Classifier",
    page_icon="🌴",
    layout="centered"
)

# Disease info for sidebar
DISEASE_INFO = """
### 🌴 About Date Palm Diseases

- **Healthy:** The palm leaf is free from visible disease symptoms.
- **White Scale:** Caused by insect pests, appears as white, powdery patches on leaves, which can weaken the plant.
- **Brown Spots:** Fungal or bacterial infection causing brown, necrotic lesions on the leaves, reducing photosynthesis and yield.

**Early detection is crucial for effective management and healthy crop yield.**
"""

# Custom messages for each class
CLASS_MESSAGES = {
    "healthy": "✅ The palm leaf appears healthy. No disease detected. Keep monitoring regularly!",
    "white_scale": "⚠️ White Scale detected! Consider pest management strategies to protect your crop.",
    "Brown_spots": "❗ Brown Spots detected! Early intervention with fungicides or pruning may be necessary."
}

# Lottie animation loader
def load_lottie(path_or_url: str):
    if path_or_url.startswith("http"):
        r = requests.get(path_or_url)
        if r.status_code != 200:
            return None
        return r.json()
    else:
        with open(path_or_url, "r") as f:
            return json.load(f)

# Agriculture Lottie animation
lottie_url = "https://assets2.lottiefiles.com/packages/lf20_3vbOcw.json"
lottie_json = load_lottie(lottie_url)

# --- Sidebar: Palm tree image and About the Project info ---
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/256/palm-tree.png",
        width=180,
      
    )
    st.markdown("""
    ## ℹ️ About the Project

    This web application uses deep learning to detect and classify common date palm leaf diseases from images.
    
    **Features:**
    - Upload an RGB image of a date palm leaf.
    - The app find the disease and segment.
    - The model predicts the disease class (Healthy, White Scale, Brown Spots).
    - Get instant feedback and disease management suggestions.

    **Goal:**  
    Help farmers and researchers quickly identify and manage date palm diseases for healthier crops and improved yield.
    """, unsafe_allow_html=True)

# --- Main page: Animation at top, then title and subtitle ---
if lottie_json:
    st_lottie(lottie_json, height=180, key="agriculture-main")

st.markdown("# 🌴 Date Palm Disease Classifier")
st.markdown("Detect and classify common date palm leaf diseases using AI.")

# Dataset and class setup
DATASET_ROOT = r"segmentation_dataset"
CLASS_NAMES = ['Brown_spots', 'healthy', 'white_scale']

# Load model
model = VGG19_CBAM(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load('vgg19_cbam_imageonly.pth', map_location=torch.device('cpu')))
#model.load_state_dict(torch.load('vgg19_cbam_imageonly.pth', map_location=torch.device('cpu'), weights_only=True))

model.eval()

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
])

def resize_image(image, width=400, height=400):
    return image.resize((width, height))

# File uploader
uploaded_file = st.file_uploader("Upload a date palm RGB image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    rgb_image = Image.open(uploaded_file).convert('RGB')
    filename = os.path.basename(uploaded_file.name)
    mask_path = None
    for class_name in CLASS_NAMES:
        possible_mask = os.path.join(DATASET_ROOT, class_name, "masks", filename)
        if os.path.exists(possible_mask):
            mask_path = possible_mask
            break

    if mask_path is not None:
        mask_image = Image.open(mask_path).convert('RGB')
        rgb_disp = resize_image(rgb_image)
        mask_disp = resize_image(mask_image)

        # Predict class
        mask_tensor = transform(mask_image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(mask_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[predicted.item()]

        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_disp, caption="🖼️ Input Image", use_container_width=True)
        with col2:
            st.image(mask_disp, caption="🎯 Predicted Mask", use_container_width=True)

        # Center the predicted class and message below the images
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; align-items: center; margin-top: 30px;">
                <span style="font-size: 22px; color: green; font-weight: bold;">
                    🟢 Predicted Class: {predicted_class}
                </span>
                <div style="margin-top: 10px; font-size: 18px;">
                    {CLASS_MESSAGES.get(predicted_class, "No additional information available.")}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("❌ Corresponding mask not found for this image.")
else:
    st.info("⬆️ Please upload a date palm RGB image to begin.")