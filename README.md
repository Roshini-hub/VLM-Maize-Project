# 🌽 Explainable Vision-Language Model for Maize Leaf Disease Detection using CLIP

## 📌 Project Overview

Maize crops face severe threats from diseases such as Northern Leaf Blight, Common Rust, and Gray Leaf Spot, leading to significant global yield losses. Traditional Convolutional Neural Networks (CNNs) offer high accuracy but rely on rigid class labels and massive labeled datasets, struggling to generalize to new variants.

This project introduces a **Multimodal Artificial Intelligence** approach using OpenAI's **CLIP (Contrastive Language-Image Pretraining)** architecture. By aligning visual symptom patterns of maize leaves with textual disease descriptions, the model enables robust **zero-shot classification**.

To bridge the gap between high-level AI and grassroots agricultural needs, this project integrates a **multilingual accessibility module** that translates diagnostic results into regional languages and synthesizes voice feedback, ensuring accessibility for farmers regardless of literacy levels or language barriers.

## ✨ Key Features

* **Zero-Shot Disease Classification:** Utilizes `openai/clip-vit-base-patch32` to match leaf images directly against semantic text prompts, eliminating the need for rigid categorical training.
* **Advanced Image Preprocessing:** Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to highlight disease lesions and Gaussian Blur to reduce background noise.
* **Multilingual Translation:** Automatically translates English disease predictions into regional languages using `googletrans`.
* **Audio Feedback (Text-to-Speech):** Converts the translated diagnostic results into an embedded audio player using `gTTS` (Google Text-to-Speech), improving accessibility for end-users.
* **Automated Data Retrieval:** Seamlessly fetches datasets using `kagglehub`.

## 🛠️ Technology Stack

* **Deep Learning Framework:** PyTorch, Torchvision
* **Foundation Model:** Hugging Face `transformers` (OpenAI CLIP)
* **Computer Vision:** OpenCV (`cv2`), Pillow (PIL)
* **Data Processing & Metrics:** NumPy, Scikit-learn
* **Accessibility:** `gTTS`, `googletrans==4.0.0-rc1`, IPython Audio
* **Visualization:** Matplotlib

## 📊 Dataset & Classes

The model evaluates maize leaves across four primary categories:

1. **Blight** (Northern Leaf Blight)
2. **Common Rust**
3. **Gray Leaf Spot**
4. **Healthy**

## 🚀 Installation & Setup

**1. Create a virtual environment (Optional but recommended)**

```bash
python -m venv venv
venv\Scripts\activate

```

**2. Install required dependencies**

```bash
python -m pip install -r require.txt ipykernel

```

## 💻 Usage

The core pipeline is implemented in a Jupyter Notebook environment in vs code.

1. Launch vs code:

```bash
code .

```

2. Open `main.ipynb` (or `main - Copy.ipynb`).
3. Run the cells sequentially. The notebook will automatically:
* Download the dataset via `kagglehub`.
* Preprocess the images (CLAHE, Blur, Resize to 224x224).
* Load the CLIP model and tokenizers.
* Run the evaluation and print the similarity scores.
* Generate the translated output and an interactive audio widget for the final prediction.



## 📈 Evaluation Metrics

The current implementation achieves highly accurate alignment between visual features and textual descriptions.

* **Overall Accuracy:** 93.8%
* **Cosine Similarity:** Utilized for exact vector matching between image embeddings and text embeddings.

**Classification Report Summary:**
| Disease Class      | Precision | Recall | F1-Score |
| :------------------| :---------| :------| :--------|
| **Blight**         | 0.86      | 0.95   | 0.90     |
| **Common Rust**    | 1.00      | 0.96   | 0.98     |
| **Gray Leaf Spot** | 0.86      | 0.74   | 0.79     |
| **Healthy**        | 1.00      | 1.00   | 1.00     |
| **Macro Average**  | **0.93**  |**0.91**| **0.92** |

## 🔮 Future Scope

* **UI Integration:** Wrapping the backend logic into a user-friendly Web App (e.g., Streamlit or Flask) or Mobile Application.
* **Prompt Engineering:** Refining the textual symptom descriptions to further improve the recall of challenging classes like Gray Leaf Spot.
* **Expanded Crop Support:** Extending the zero-shot capabilities to other vital crops like wheat and rice.

## 📚 References

* Hugging Face Transformers Documentation: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
