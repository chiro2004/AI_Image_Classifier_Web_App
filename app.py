import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_size = 48

model = tf.keras.models.load_model("AIGeneratedModel.h5")

st.title("AI Image Classifier Web App")       
        
img = st.file_uploader("Insert an image")

if img and st.button("Run"):
    image = Image.open(img)
    st.image(img)
    image = ImageOps.fit(image, (48,48), Image.Resampling.LANCZOS)
    img_array = img_to_array(image)
    new_arr = img_array/255
    test = []
    test.append(new_arr)
    test = np.array(test)
    y = model.predict(test)
    if y[0] <= 0.5:
        st.write("The above image is REAL.")
    else:
        st.write("The given image is AI GENERATED.")
    