import streamlit as st
from pytube import YouTube
from ultralytics import YOLO
import os
import cv2
from tqdm import trange
import torch
import numpy as np

def process(video_path):
    torch.cuda.set_device(0)
    print("Torch cuda is available: ", torch.cuda.is_available())
    video_path = video_path.replace('\\', '/')  

    if not os.path.exists('./results'):
        os.makedirs('./results')
    output_video_path = f'./results/process_{os.path.basename(video_path)}'

    cap = cv2.VideoCapture(video_path)
    output_fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    output_video = cv2.VideoWriter(output_video_path, fourcc, output_fps, output_size)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    model = YOLO("yolov8n.pt")
    progress_bar = st.progress(0)

    for index in trange(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        progress_bar.progress(index / frame_count)
        results = model.predict(frame, verbose=False)
        result = results[0]
        frame = result.plot()
        
        count = 0
        class_id = result.boxes.cls.cpu().numpy()
        count = np.count_nonzero(class_id == 0)
        text = f'Number of people: {count}'
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        output_video.write(frame)
    progress_bar.progress(100)
    
    output_video.release()
    cap.release()
    st.video(output_video_path)
        
    
if __name__ == "__main__":
    
    st.title("Video Analytics System")
    choice = st.radio("Upload video from", ["URL link", "Browse"])
    save_file = os.getcwd()

    if choice == "Browse":
        uploaded_file = st.file_uploader("Choose a video...")
        
        if uploaded_file:
            # Specify the folder where you want to save the uploaded video
            
            # Get the file name from the uploaded file
            file_name = uploaded_file.name
            file_path = os.path.join(save_file, file_name)
            
            # Save the uploaded file to the specified folder
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())

            
            if st.button("Analyze", type="primary"):
                process(file_path)
    else:
        url = st.text_input('Enter the URL link')
        if st.button("Analyze", type="primary"):
            yt = YouTube(url)
            stream = yt.streams.get_highest_resolution()
            download_path = os.path.join(save_file, stream.default_filename)
            stream.download(output_path=save_file)
            process(download_path)