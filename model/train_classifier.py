import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torchvision.models import resnet50, ResNet50_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained ResNet18
#from torchvision.models import ResNet18_Weights
#resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#resnet.fc = torch.nn.Identity()
weights = ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder("D:/internshala/medical_image_classifier_package/medical_vs_nonmedical", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

X, y = [], []
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = resnet(imgs).cpu().numpy()
        X.extend(feats)
        y.extend(labels.numpy())

# Train classifier
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
#clf = LogisticRegression(max_iter=1000)
#clf.fit(X_train, y_train)

from xgboost import XGBClassifier

clf = XGBClassifier(eval_metric='logloss')
clf.fit(X, y)



# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy on test set: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=dataset.classes))


# Save model and label map
os.makedirs("D:/internshala/medical_image_classifier_package/app", exist_ok=True)
pickle.dump(clf, open("D:/internshala/medical_image_classifier_package/app/classifier.pkl", "wb"))
pickle.dump(dataset.classes, open("D:/internshala/medical_image_classifier_package/app/label_map.pkl", "wb"))

print("✅ Classifier trained and saved at D:/internshala/medical_image_classifier_package/app")
