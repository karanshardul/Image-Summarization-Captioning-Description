# #Image summarization

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

def generate_image_captions(image):
    processor, model = load_blip_model()

    # Conditional image captioning (optional)
    text = "a photography of"  # Adjust the prompt if needed
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)

    # Unconditional image captioning
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

    return conditional_caption, unconditional_caption

st.set_page_config("Image Summarizor", ":turkey:", layout="wide")

filename = st.file_uploader('Upload an image')

if filename is not None:
    # if filename.endswith(".jpeg") or filename.endswith(".png"):  # Check for image extensions
        # Open the image in a streaming fashion
        image_bytes = filename.getvalue()
        raw_image = Image.open(io.BytesIO(image_bytes))

        conditional_caption, unconditional_caption = generate_image_captions(raw_image)

        st.write(f"Conditional caption: {conditional_caption}")
        st.write(f"Unconditional caption: {unconditional_caption}")