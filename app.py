import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
import subprocess
import shutil
import atexit

model = YOLO("./Staff_Detection/Version3/weights/best.pt")

def convert_avi_to_mp4(input_path, output_path):
    ffmpeg_command = [
        'ffmpeg', '-i', input_path, '-vcodec', 'libx264', output_path
    ]
    subprocess.run(ffmpeg_command, check=True)

def cleanup_runs_directory():
    shutil.rmtree("./runs/detect")
    os.makedirs("./runs")

# Streamlit app configuration
st.set_page_config(page_title="Staff Detection")
atexit.register(cleanup_runs_directory)

st.title("Staff Detection")
st.write("Upload an image/video to detect staff in the image.")
st.write("Supported file types: jpg, jpeg, png, mp4")

# Initialize the counter for prediction runs if not already set
if 'i' not in st.session_state:
    st.session_state.i = 1

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            st.image(image, caption="Uploaded Image")

            # Run model prediction and save results in text format
            results = model.predict(source=image, save_txt=True, iou=0.7, conf=0.5)
            result_img = results[0].plot()
            result_img_rgb = Image.fromarray(result_img[..., ::-1])
            # Display the processed image
            st.image(result_img_rgb, caption="Detected Image")
            st.success("Image processing complete!")

            if st.session_state.i < 2:
                result_path = "./runs/detect/predict/labels/image0.txt"
            else:
                result_path = f"./runs/detect/predict{st.session_state.i}/labels/image0.txt"
            
            if os.path.exists(result_path):
                detections = []
                with open(result_path, "r") as f:
                    for line in f:
                        cls, x_center, y_center, width, height = map(float, line.strip().split())
                        detections.append([int(cls), x_center, y_center, width, height])
                
                if detections:
                    detection_df = pd.DataFrame(detections, columns=["Class ID", "X Center", "Y Center", "Width", "Height"])
                    st.write("Detected objects and coordinates:", detection_df)
                else:
                    st.write("No staff detected.")
            else:
                st.write("Label file not found. Please try again.")

            st.session_state.i += 1

    elif uploaded_file.type == "video/mp4":
        if st.session_state.i < 2:
            result_path = "./runs/detect/predict"
        else:
            result_path = f"./runs/detect/predict{st.session_state.i}"
            
        print(result_path)

        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        with st.spinner("Processing video... This may take a while."):
            # Run model prediction on the video
            results = model.predict(source=tfile.name, save=True, save_txt=True, iou=0.7, conf=0.6)
            
        input_video = os.path.join(result_path, os.path.basename(tfile.name).replace(".mp4", ".avi"))
        output_video = os.path.join(result_path, os.path.basename(tfile.name).replace(".avi", ".mp4"))

        # Convert and display
        convert_avi_to_mp4(input_video, output_video)
        st.video(output_video)
        st.success("Video processing complete!")

        # Define path for the label files
        label_directory = os.path.join(result_path, 'labels')
        if os.path.exists(label_directory):
            frame_detections = []
            for label_file in os.listdir(label_directory):
                if label_file.endswith('.txt'):
                    frame_number = int(label_file.split('_')[-1].replace(".txt", ""))  # Extract frame number from file name
                    with open(os.path.join(label_directory, label_file), "r") as f:
                        for line in f:
                            cls, x_center, y_center, width, height = map(float, line.strip().split())
                            frame_detections.append([frame_number, int(cls+1), x_center, y_center, width, height])

            # Convert to DataFrame for display
            if frame_detections:
                frame_detection_df = pd.DataFrame(frame_detections, columns=["Frame", "Class ID", "X Center", "Y Center", "Width", "Height"])
                st.write("Detected objects by frame and coordinates:", frame_detection_df)
            else:
                st.write("No staff detected in the video.")
        else:
            st.write("No label files found for the video.")

        # Clean up temporary files
        os.remove(tfile.name)
        
        # Increment `i` after each video prediction
        st.session_state.i += 1
