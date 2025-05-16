import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from dotenv import load_dotenv
import os
import google.generativeai as genai
from deep_translator import GoogleTranslator
from langdetect import detect

# ✅ Load plant disease model
@st.cache_resource
def load_saved_model():
    return load_model("plant-disease.h5")

model = load_saved_model()

# ✅ Plant disease labels
categories = [
    'Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Healthy Apple Leaf',
    'Corn Gray Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight', 'Healthy Corn Leaf',
    'Potato Early Blight', 'Potato Late Blight', 'Healthy Potato Leaf',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 'Healthy Tomato Leaf'
]

# ✅ Load Gemini AI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-1.5-flash")
chat = model_gemini.start_chat(history=[])

# ✅ Translator
def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_to_hindi(text):
    return GoogleTranslator(source='en', target='hi').translate(text)

# ✅ Gemini prompt
def get_response(input_text):
    try:
        prompt = f"""
        You are an agricultural assistant AI. When given a plant disease name, provide the following in bullet points:

        **Disease Name:** <Definition>  
        **Affected Plant/Part:** <Which plant/part it affects>  
        **Symptoms:** <Signs to look for>  
        **Causes:** <Fungal/Bacterial/Environmental causes>  
        **Treatment & Prevention:** <Methods to treat and prevent>  
        **Farmer Advice:** <Tips for farmers and when to consult an agronomist>

        The input is: "{input_text}"
        """
        response = chat.send_message(prompt, stream=True)
        return response
    except Exception as e:
        st.error(f"Error fetching response: {e}")
        return []

# ✅ UI Setup
st.title("🌾 AgriBot - Plant Disease Detection & Advice")
st.write("Choose your input method: 📷 Image | 📝 Text")

option = st.radio("Select Input Method:", ("Text Input", "Image Upload"))

# 1️⃣ TEXT INPUT
if option == "Text Input":
    user_input = st.text_input("Ask your plant-related question (in Hindi or English):", placeholder="e.g. टमाटर की पत्तियाँ पीली क्यों हो रही हैं?")
    submit = st.button("Get Advice")

    if submit and user_input:
        with st.spinner("Detecting language and analyzing..."):
            detected_lang = detect(user_input)
            translated_input = translate_to_english(user_input)

            response_chunks = get_response(translated_input)
            if response_chunks:
                response_en = "".join([chunk.text for chunk in response_chunks])
                response = translate_to_hindi(response_en) if detected_lang == "hi" else response_en
                st.write("### 🌿 Expert Advice:")
                st.info(response)
            else:
                st.warning("Unable to fetch response.")

# 2️⃣ IMAGE UPLOAD
elif option == "Image Upload":
    uploaded_file = st.file_uploader("Upload a plant leaf image:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Plant Leaf", use_column_width=True)

        img = load_img(uploaded_file, target_size=(224, 224))  # ✅ resized correctly
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        predicted_class_index = np.argmax(pred, axis=1)[0]
        predicted_class_name = categories[predicted_class_index]

        st.success(f"🌿 Detected Disease: **{predicted_class_name}**")

        with st.spinner("Getting expert advice..."):
            response_chunks = get_response(predicted_class_name)
            if response_chunks:
                response_en = "".join([chunk.text for chunk in response_chunks])
                response_hi = translate_to_hindi(response_en)

                st.write("### 📗 Expert Advice (English):")
                st.info(response_en)

                st.write("### 📘 किसान के लिए जानकारी (Hindi):")
                st.info(response_hi)
            else:
                st.warning("Could not retrieve additional details.")
