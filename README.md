# Crop Disease Detection with RAG Chatbot

This repository contains the internship project titled **"Crop Disease Detection with RAG (Retrieval-Augmented Generation) Chatbot"**, developed to assist farmers and agricultural professionals in identifying crop diseases using machine learning and providing context-aware responses through an AI-powered, multilingual chatbot.

---

## Project Overview

The project integrates two core components:

1. **Crop Disease Detection Model** – A computer vision-based model trained to classify plant diseases from crop leaf images.
2. **RAG Chatbot** – A multilingual Retrieval-Augmented Generation chatbot that provides accurate and relevant responses based on a knowledge base and user queries.

Users can upload crop images via a **Streamlit web interface**, receive predictions about potential diseases, and ask follow-up questions in **multiple languages** through the chatbot for treatment advice, prevention tips, and more.

---

##Model Details

- A **Convolutional Neural Network (CNN)** model was trained on a curated dataset of diseased crop leaves.
- The trained model (`.h5` file) has **not been included** in this repository due to size limitations or confidentiality.
- The Streamlit app for interfacing with the model is included and fully functional.
- You can easily replace the model path in `app.py` with your own trained `.h5` file to run it locally.

---

## RAG Chatbot Features

- Utilizes **Retrieval-Augmented Generation (RAG)** for intelligent Q&A.
- Integrates with large language models (e.g., OpenAI, Cohere).
- Supports **multilingual inputs and outputs**, making it accessible to users from different linguistic backgrounds.
- Built-in context handling enables relevant and helpful responses based on uploaded crop disease and user intent.
- Only the **API key needs to be updated** in the chatbot script to activate full functionality.

---

## Project Structure

Crop_Disease_Detection/
│
├── chroma_store/ # Local vector database for chatbot context
├── rag-api/ # Code for the RAG chatbot API
├── Vector_Store/ # Pre-generated embeddings for RAG
│
├── .env # Environment variables (e.g., API keys)
├── api/ # Supporting text files or API data
├── app.py # Streamlit app for image prediction and chatbot UI
│
├── plant-disease.h5 # Final trained model file (also not uploaded to GitHub)
├── Plant_Disease_Prediction_CNN_Image_Classification.ipynb # Model training notebook
├── frontend/ # React-based UI (under development)
│ ├── public/
│ ├── src/
│ ├── package.json
│ ├── package-lock.json
│ └── .gitignore
│
├── requirements.txt # Python dependencies


> 🔧 **Note:** The React-based frontend app (`frontend/`) is currently in progress and will be finalized once the RAG chatbot backend integration is fully functional.

---

##  Model Description

- The `.h5` file is a trained **Convolutional Neural Network (CNN)** model for identifying plant leaf diseases.
- Due to size limitations, this file is **not included in the GitHub repository**.
- You can train your own model using the provided Jupyter Notebook or request the file for testing.
- Model predictions are served through a **Streamlit app** (`app.py`).

---

## RAG Chatbot Description

- The chatbot is implemented inside the `rag-api/` directory.
- It uses **ChromaDB** (`chroma_store/`) as a local vector store.
- Multilingual support is enabled for both input and output (you can ask questions in different languages).
- API key for OpenAI (or your chosen provider) should be added in the `.env` file.

---

## Setup Instructions

### 1. Clone the repository:
```bash
git clone https://github.com/punyatagupta-disys/Crop_Disease_Detection.git
cd Crop_Disease_Detection

Internship Notes
Project developed as part of an internship at DISYS.

It showcases the real-world potential of combining AI in agriculture with human-centric NLP technologies.

The project demonstrates both technical proficiency and practical application design.

Future Enhancements
Enable voice input/output for chatbot interaction.

Deploy to cloud for remote farmer access.

Incorporate real-time image capture via drone/mobile.

Add Grad-CAM visualization to explain predictions.
