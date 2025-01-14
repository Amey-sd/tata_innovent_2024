from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
import subprocess
from document import prepare_report_data

file_path = os.path.dirname(__file__)
model_path4 = os.path.join(file_path, 'models', 'Yolov8', 'Large XL', 'weights', 'best.pt')
model4 = YOLO(model_path4)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
IMG_FOLDER = 'static/images'
DOC_FOLDER = 'doc'
csv = os.path.join(DOC_FOLDER, 'data.csv')
template = os.path.join(DOC_FOLDER, 'template.docx')
output = os.path.join(DOC_FOLDER, 'Report.docx')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/image')
def index():
    return render_template('image.html')

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
    
    if model_choice == "model4":
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

# Loop through each report
    for r in report:
        # Display the image as is
        if hasattr(r, 'image') and r.image is not None:
            # Assuming the original image is accessible via r.image
            original_image = r.image
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Original Image")
            plt.show()
        else:
            print("No image found in the report.")

        if r.boxes is not None:
            # Convert results to list for further analysis
            class_ids = r.boxes.cls.cpu().numpy().tolist()
            class_ids = [int(cls_id) for cls_id in class_ids]
            class_names = [model.names[int(cls_id)] for cls_id in class_ids]

            # Calculate mask areas
            masks = r.masks.data if r.masks is not None else None  # Access the raw mask data
            mask_areas = []

            if masks is not None:
                for mask in masks:
                    # Convert each mask to binary format and calculate area
                    binary_mask = mask.cpu().numpy()  # Get the numpy array representation of the mask
                    area = cv2.countNonZero(binary_mask)  # Count non-zero pixels in the mask
                    mask_areas.append(area)

            print("Class IDs:", class_ids)
            print("Class Names:", class_names)
            print("Mask Areas:", mask_areas)  # List of areas for each detected object

            
            data = {
                "Class ID": class_ids,
                "Damage Type": class_names,
                "Area": mask_areas
            }
            df = pd.DataFrame(data)
            df.to_csv(csv, index=False)

            prepare_report_data(csv, template, output)
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
    if model_choice == "model4":
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
    processed_video_name = f'{filename}'
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

        frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
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

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Parse incoming JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract and decode the image
        image_data = data.get("image")
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Remove the data URL prefix and decode
        image_base64 = image_data.split(",")[1] if "," in image_data else image_data
        image_bytes = base64.b64decode(image_base64)

        # Convert bytes to a NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Retrieve the selected model
        model_choice = data.get('model')
        if not model_choice:
            return jsonify({'error': 'No model choice provided'}), 400

        # Select the appropriate model
        if model_choice == "model4":
            model = model4
        else:
            return jsonify({'error': 'Invalid model choice'}), 400

        # Run YOLO inference
        results = model.predict(frame, stream=False)
        if not results:
            return jsonify({'error': 'No results from model'}), 500

        result = results[0]

        # Annotate frame with predictions
        annotated_frame = result.plot()

        # Encode annotated frame as base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        response_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"image": f"data:image/jpeg;base64,{response_base64}"})
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)
    
@app.route('/doc/<filename>')
def processed_doc(filename):
    if filename == 'Report.docx':
        return send_from_directory(DOC_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)