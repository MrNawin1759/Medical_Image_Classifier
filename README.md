# Medical_Image_Classifier

📌 Project Overview
This project is a deep learning pipeline that classifies images as either medical or non-medical. The pipeline can accept inputs in the form of:

Uploaded PDFs (extracts and classifies embedded images)

Uploaded images

Webpage URLs (scrapes and classifies images from the page)

A real-time interactive frontend is built using Streamlit, and backend classification is done using a combination of ResNet50 (for feature extraction) and XGBoost (for classification).

🧠 Approach and Reasoning
🔹 Feature Extraction
A pretrained ResNet50 model (from torchvision) is used to extract 2048-dimensional feature vectors for each image.

The final classification layer of ResNet50 is replaced with an identity layer (resnet.fc = torch.nn.Identity()), allowing us to use it as a feature encoder.

🔹 Image Augmentation
During training, random transformations like resizing, flipping, rotation, and color jitter were applied to improve robustness and generalization.

🔹 Classifier
Extracted features are used to train a supervised classifier (XGBoost).

XGBoost was chosen over simpler models (like logistic regression) for its better handling of small datasets, regularization, and interpretability.

🎯 Accuracy Results
Metric	Value
Accuracy	100.00%
Precision	1.00 (both classes)
Recall	1.00 (both classes)
F1-score	1.00 (both classes)
Test Size	20% of full dataset

These results were obtained on a small, manually labeled validation set containing 15 test images (9 medical + 6 non-medical). While the results are perfect, a larger and more diverse dataset would be required to validate generalization.

⚙️ Performance and Efficiency Considerations
✅ Real-time Predictions
Feature extraction is GPU-accelerated (if CUDA is available).

Predictions using XGBoost are fast (~<100ms per image on CPU).

✅ Lightweight App
Uses Streamlit for rapid prototyping and deployment.

PDF image extraction via PyMuPDF is efficient and supports most formats.

Web scraping is lightweight using BeautifulSoup.

❗ Potential Bottlenecks
Web URL image extraction can be slow or blocked due to external server limitations.

Large PDFs or websites with many images may slightly impact response time.


