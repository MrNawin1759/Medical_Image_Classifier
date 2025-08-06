import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
import fitz  # PyMuPDF
from PIL import Image
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import os
import  numpy as np
from torchvision.models import resnet50, ResNet50_Weights
# --- Load classifier and label map ---
clf = pickle.load(open("D:/internshala/medical_image_classifier_package/app/classifier.pkl", "rb"))
label_map = pickle.load(open("D:/internshala/medical_image_classifier_package/app/label_map.pkl", "rb"))

# --- Load feature extractor ---
device = "cuda" if torch.cuda.is_available() else "cpu"
#resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#resnet.fc = torch.nn.Identity()


import torchvision.models as models
weights = ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def predict_image(img: Image.Image):
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_t).numpy()
    probs = clf.predict_proba(features)[0]
    pred_index = np.argmax(probs)
    pred_label = label_map[pred_index]
    confidence = probs[pred_index]
    return pred_label, confidence


def extract_features(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img).cpu().numpy()
    return features

def classify_image(img):
    features = extract_features(img)
    pred = clf.predict(features)[0]
    return label_map[pred]

def extract_images_from_pdf(pdf_file):
    images = []
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            images.append(image)
    return images

def extract_images_from_url(url):
    images = []
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for img_tag in soup.find_all("img"):
        img_url = img_tag.get("src")
        try:
            if img_url.startswith("http"):
                response = requests.get(img_url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(image)
        except Exception:
            continue
    return images

# --- Streamlit UI ---
st.title("🩺 Medical vs Non-Medical Image Classifier")

input_mode = st.selectbox("Choose Input Type:", ["Upload PDF", "Upload Image(s)", "Enter URL"])

if input_mode == "Upload PDF":
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf_file:
        st.write("🔍 Extracting images...")
        images = extract_images_from_pdf(pdf_file)
        st.success(f"Found {len(images)} image(s) in PDF.")
        for img in images:
            st.image(img, width=300)
            label, confidence = predict_image(img)
            st.success(f"🧠 Prediction: **{label}** with **{confidence * 100:.2f}%** confidence")

elif input_mode == "Upload Image(s)":
    image_files = st.file_uploader("Upload image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        st.image(img, width=300)
        label, confidence = predict_image(img)
        st.success(f"🧠 Prediction: **{label}** with **{confidence * 100:.2f}%** confidence")


elif input_mode == "Enter URL":
    url = st.text_input("Enter a URL")
    if st.button("Fetch and Classify"):
        if url:
            st.write("🔍 Extracting images from URL...")
            images = extract_images_from_url(url)
            st.success(f"Found {len(images)} image(s) on page.")
            for img in images:
                st.image(img, width=300)
                label, confidence = predict_image(img)
                st.success(f"🧠 Prediction: **{label}** with **{confidence * 100:.2f}%** confidence")

