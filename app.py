# --- START OF CORRECTED app.py ---
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import os

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Our Family Fruit Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.header("Image Config")
    confidence_threshold = st.slider("Select Model Confidence", 0.0, 1.0, 0.25, 0.01)

# --- Main Page ---
st.title("🍎 Our Family Fruit Detection using YOLOv8")
st.write("Upload an image and our YOLOv8 model will detect the fruits in it. This model can detect apples, bananas, and oranges.")


# --- Caching the Model ---
@st.cache_resource
def load_model(model_path):
    """Loads a YOLOv8 model from the specified path."""
    model = YOLO(model_path)
    return model

# --- Load the trained YOLOv8 model (THE CORRECT WAY) ---
try:
    # Construct the absolute path to the model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'best.pt')

    # Check if the model file exists before loading
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Make sure 'best.pt' is in your GitHub repository.")
        st.stop()

    model = load_model(model_path)

except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# --- Image Upload and Processing ---
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read the uploaded file as an image
    image = Image.open(uploaded_file)

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Perform detection when the button is clicked
    if st.button("Detect Fruits"):
        with st.spinner("Detecting..."):
            # Run the model on the image
            results = model.predict(source=image, conf=confidence_threshold)

            # The .plot() method returns a BGR numpy array with the detections drawn on it
            result_plot = results[0].plot()

            # Convert the BGR array to RGB for display in Streamlit
            result_image_rgb = result_plot[:, :, ::-1]

            with col2:
                st.image(result_image_rgb, caption="Detected Image", use_container_width=True)

            # Display detected classes and their counts
            st.subheader("Detected Objects:")
            names = model.names
            detected_counts = {}
            for r in results:
                for c in r.boxes.cls:
                    class_name = names[int(c)]
                    detected_counts[class_name] = detected_counts.get(class_name, 0) + 1

            if detected_counts:
                for class_name, count in detected_counts.items():
                    st.write(f"- **{class_name.capitalize()}**: {count}")
            else:
                st.write("No fruits detected.")

else:
    st.info("Please upload an image file to get started.")

# --- END OF CORRECTED app.py ---
