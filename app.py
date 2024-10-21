# streamlit_app.py
'''
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model (ensure model.keras is in the same directory)
model = load_model('model.keras')

# Constants
IMAGE_SIZE = (224, 224)

# Define the predict_and_recommend function
def predict_and_recommend(model, uploaded_images):
    """Predicts the best image and recommends which ones to delete."""
    best_image = None
    delete_images = []
    best_score = 0

    for img in uploaded_images:
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        score = model.predict(img_array)[0][0]

        if score > best_score:
            best_score = score
            best_image = img
        else:
            delete_images.append(img)

    return best_image, delete_images, best_score

# Streamlit Layout
st.set_page_config(page_title="Image Quality Assessment", layout="wide")
st.title("Image Quality Assessment System")
st.markdown("### Manage Digital Storage with Optimal Image Recommendations")

st.write("Upload multiple images to analyze their quality. The best-quality image will be displayed along with a recommendation to delete lower-quality images.")

# File uploader for multiple images
uploaded_files = st.file_uploader(
    "Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
)

if uploaded_files:
    # Convert uploaded files to PIL images
    images = [Image.open(file).resize(IMAGE_SIZE) for file in uploaded_files]

    # Perform prediction and recommendation
    best_image, delete_images, best_score = predict_and_recommend(model, images)

    # Display the accuracy score of the best image
    st.subheader(f"Best Image Quality Score: {best_score:.2f}")

    # Display the best image
    st.image(best_image, caption="Best Quality Image", use_column_width=True)

    # Display images recommended for deletion
    st.subheader("Images to Delete")
    for img in delete_images:
        st.image(img, use_column_width=True)

else:
    st.info("Please upload at least one image to proceed.")
'''

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('model.keras')

# Constants
IMAGE_SIZE = (224, 224)

# Initialize session state if not already set
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "scores" not in st.session_state:
    st.session_state.scores = []
if "delete_indices" not in st.session_state:
    st.session_state.delete_indices = set()

# Define the function to predict and recommend images
def predict_and_recommend(model, uploaded_images):
    """Predicts quality scores and identifies the best image."""
    scores = []
    best_image_index = 0
    best_score = 0

    for idx, img in enumerate(uploaded_images):
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        score = model.predict(img_array)[0][0]
        scores.append((img, score))

        if score > best_score:
            best_score = score
            best_image_index = idx

    best_image = uploaded_images[best_image_index]
    return best_image, scores

# Streamlit Layout
st.set_page_config(page_title="Image Quality Assessment", layout="wide")
st.title("Image Quality Assessment System")
st.markdown("### Manage Digital Storage with Optimal Image Recommendations")

st.write(
    "Upload multiple images to analyze their quality. "
    "The best-quality image will be displayed, and other images will be recommended for deletion."
)

# File uploader for multiple images with drag and drop support
uploaded_files = st.file_uploader(
    "Upload Images", 
    accept_multiple_files=True, 
    type=["jpg", "jpeg", "png"],
    help="Drag and drop files here or click to select multiple files. Limit 200MB per file."
)

if uploaded_files:
    # Convert uploaded files to PIL images
    st.session_state.uploaded_images = [
        Image.open(file).resize(IMAGE_SIZE) for file in uploaded_files
    ]

    # Perform prediction and recommendation
    best_image, st.session_state.scores = predict_and_recommend(
        model, st.session_state.uploaded_images
    )

# Display the accuracy scores for all images
if st.session_state.uploaded_images:
    st.subheader("Image Quality Scores:")
    score_cols = st.columns(len(st.session_state.scores))

    for i, (img, score) in enumerate(st.session_state.scores):
        if i not in st.session_state.delete_indices:
            with score_cols[i]:
                st.image(img, caption=f"Score: {score:.2f}", width=100)

    # Display the best image
    best_image_score = next(
        score for img, score in st.session_state.scores if img == best_image
    )
    st.subheader(f"Best Image (Score: {best_image_score:.2f})")
    st.image(best_image, caption="Best Quality Image", width=300)

    # Display images recommended for deletion
    st.subheader("Images Recommended for Deletion")
    delete_cols = st.columns(len(st.session_state.uploaded_images) - 1)

    for i, (img, _) in enumerate(st.session_state.scores):
        if i != 0 and i not in st.session_state.delete_indices:
            with delete_cols[i - 1]:
                st.image(img, use_column_width=True)
                if st.button(f"Delete Image {i}", key=f"delete_{i}"):
                    st.session_state.delete_indices.add(i)

    # Remove images marked for deletion
    st.session_state.uploaded_images = [
        img for idx, img in enumerate(st.session_state.uploaded_images)
        if idx not in st.session_state.delete_indices
    ]
else:
    st.info("Please upload at least one image to proceed.")
