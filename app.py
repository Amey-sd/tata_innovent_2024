from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from flask_socketio import SocketIO
from PIL import Image
import subprocess

file_path = os.path.dirname(__file__)
model_path1 = os.path.join(file_path, 'models', 'Yolov8', 'Mini', 'weights', 'best.pt')
model_path2 = os.path.join(file_path, 'models', 'Yolov8', 'Medium', 'weights', 'best.pt')
model_path3 = os.path.join(file_path, 'models', 'Yolov8', 'Large', 'weights', 'best.pt')
model_path4 = os.path.join(file_path, 'models', 'Yolov8', 'Large XL', 'weights', 'best.pt')
model1 = YOLO(model_path1)
model2 = YOLO(model_path2)
model3 = YOLO(model_path3)
model4 = YOLO(model_path4)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
IMG_FOLDER = 'static/images'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400
    
    file = request.files['file']
    model_choice = request.form.get('model')
    
    if model_choice == "model1":
        model = model1
    elif model_choice == "model2":
        model = model2
    elif model_choice == "model3":
        model = model3
    elif model_choice == "model4":
        model = model4
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # YOLO Prediction
    results = model.predict(img, conf=0.2)
    img_with_boxes = results[0].plot()  # Matplotlib array

    # Convert Matplotlib image to send via Flask
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_with_boxes)
    ax.axis('off')
    canvas = FigureCanvas(fig)
    img_io = io.BytesIO()
    canvas.print_png(img_io)
    img_io.seek(0)

    # Convert the image to RGB (to avoid RGBA error when saving as JPEG)
    img_pil = Image.open(img_io).convert('RGB')

    # Save the processed image in the folder
    processed_filepath = os.path.join(PROCESSED_FOLDER, filename)
    img_pil.save(processed_filepath, format='JPEG')  # Save as JPEG

    report = model.track(img, conf=0.2)  # Replace with your image path

    for r in report:
        if r.boxes is not None:
            # Convert results to list for further analysis
            class_ids = r.boxes.cls.cpu().numpy().tolist()
            class_names = [model.names[int(cls_id)] for cls_id in class_ids]

            # Calculate mask areas
            masks = r.masks.data  # Access the raw mask data
            mask_areas = []

            if masks is not None:
                for mask in masks:
                    # Convert each mask to binary format and calculate area
                    binary_mask = mask.cpu().numpy()  # Get the numpy array representation of the mask
                    area = cv2.countNonZero(binary_mask)  # Count non-zero pixels in the mask
                    mask_areas.append(area)

            #print("Object IDs:", object_ids)  # type <class 'list'>
            print("Class IDs:", class_ids)
            print("Class Names:", class_names)
            print("Mask Areas:", mask_areas)  # List of areas for each detected object 
        else:
            print("No boxes detected in this report.")

# Update JSON response
    return jsonify({
        #'object_ids': object_ids,
        'class_ids': class_ids,
        'class_names': class_names,
        'mask_areas': mask_areas,
        'image_url': f'/static/processed/{filename}'
    })

def reencode_video(input_path, output_path):
    """Re-encode video to H.264 using FFmpeg."""
    command = [
        'ffmpeg',
        '-i', input_path,         # Input file
        '-vcodec', 'libx264',     # Video codec: H.264
        '-acodec', 'aac',         # Audio codec: AAC
        output_path               # Output file
    ]
    subprocess.run(command, check=True)

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file found in the request'}), 400
    
    file = request.files['video']
    model_choice = request.form.get('model')
    print(f"Model choice received: {model_choice}")

    # Select the model
    if model_choice == "model1":
        model = model1
    elif model_choice == "model2":
        model = model2
    elif model_choice == "model3":
        model = model3
    elif model_choice == "model4":
        model = model4
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    # Save the video file
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return jsonify({'error': 'Invalid video file'}), 400

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create output path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Initial codec for processing
    processed_video_name = f'processed_{filename}'
    processed_video_path = os.path.join(PROCESSED_FOLDER, processed_video_name)
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
    
    # Process each frame
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Run model prediction
        results = model.predict(frame, conf=0.2)
        frame_with_boxes = results[0].plot()  # Annotated frame as numpy array
        frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)  # Convert if needed

        out.write(frame_with_boxes)

    # Release resources
    cap.release()
    out.release()

    # Re-encode the processed video to H.264
    reencoded_video_name = f'reencoded_{filename}'
    reencoded_video_path = os.path.join(PROCESSED_FOLDER, reencoded_video_name)
    try:
        reencode_video(processed_video_path, reencoded_video_path)
        os.remove(processed_video_path)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'Failed to re-encode video', 'details': str(e)}), 500

    # Return success response
    return jsonify({
        'message': 'Video processed and re-encoded successfully',
        'processed_video_url': f'/static/processed/{reencoded_video_name}'
    }), 200


def generate_frames(model):
    # Try initializing the camera from index 0 to 5
    camera = None
    for i in range(6):  # Check camera indices 0 to 5
        camera = cv2.VideoCapture(i)
        if camera.isOpened():
            print(f"Camera found at index {i}")
            break
    if camera is None or not camera.isOpened():
        print("Error: No camera available.")
        return

    while True:
        success, frame = camera.read()  # Read the frame from the camera
        if not success:
            print("Failed to grab frame.")
            break

        # Run inference on the current frame
        results = model.predict(frame, stream=False)

        # Access the first result from the list
        result = results[0]

        # Plot the predictions on the frame
        annotated_frame = result.plot()

        # Encode the annotated frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            print("Error: Unable to encode frame.")
            break
        frame_bytes = buffer.tobytes()  # Convert frame to bytes

        # Yield the frame as multipart response for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

@app.route('/video_feed')
def video_feed():
    model_choice = request.args.get('model')

    # Select the model based on the user's choice
    if model_choice == "model1":
        model = model1
    elif model_choice == "model2":
        model = model2
    elif model_choice == "model3":
        model = model3
    elif model_choice == "model4":
        model = model4
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    # Start video feed and generate frames
    return Response(generate_frames(model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
