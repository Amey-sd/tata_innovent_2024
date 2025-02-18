from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
from document import prepare_report_data

file_path = os.path.dirname(__file__)
model_path4 = os.path.join(file_path, 'models', 'Yolov8', 'Large XL', 'weights', 'best.pt')
model4 = YOLO(model_path4)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
IMG_FOLDER = 'static/images'
DOC_FOLDER = 'doc'
LOG_FILE = os.path.join(DOC_FOLDER, 'defect_log.csv')
csv = os.path.join(DOC_FOLDER, 'data.csv')
template = os.path.join(DOC_FOLDER, 'template.docx')
output = os.path.join(DOC_FOLDER, 'Report.docx')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)

@app.route('/about')
def about():
    return render_template('about.html')

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

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/features')
def features():
    return render_template('features.html')

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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        # Display the image as is
        if hasattr(r, 'image') and r.image is not None:
            # Assuming the original image is accessible via r.image
            original_image = r.image
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Original Image")
            plt.show()
        else:
            print("No Image found")
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

            #print("Object IDs:", object_ids)  # type <class 'list'>
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
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Open the video file
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({'error': 'Invalid video file'}), 400

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create output path
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    output_video_name = f'{filename}'
    output_video_path = os.path.join(PROCESSED_FOLDER, output_video_name)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Open log file for writing defects
    with open(LOG_FILE, 'w') as log_file:
        log_file.write("Frame Number, Defect Type, Mask Area\n")  # Header for log file

        # Frame processing loop
        for frame_number in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Run model prediction
            results = model.predict(frame, conf=0.2)

            # Extract class IDs and names from results
            class_ids = results[0].boxes.cls.cpu().numpy().tolist()
            class_ids = [int(cls_id) for cls_id in class_ids]
            class_names = [model.names[int(cls_id)] for cls_id in class_ids]

            masks = results[0].masks.data if results[0].masks is not None else None  # Access the raw mask data
            mask_areas = []

            if masks is not None:
                for mask in masks:
                    # Convert each mask to binary format and calculate area
                    binary_mask = mask.cpu().numpy()  # Get the numpy array representation of the mask
                    area = cv2.countNonZero(binary_mask)  # Count non-zero pixels in the mask
                    mask_areas.append(area)

                    # Write to log file for each detected defect (assuming one area per defect)
                    defect_type = class_names[len(mask_areas) - 1]  # Get corresponding defect type
                    log_file.write(f"{frame_number}, {defect_type}, {area}\n")

            frame_with_masks = results[0].plot()  # Annotated frame as numpy array
            out.write(frame_with_masks)

    # Release resources
    cap.release()
    out.release()
    
    # Return success response
    return jsonify({
        'message': 'Video processed successfully',
        'processed_video_url': f'/static/processed/{output_video_name}',
        'defect_log_url': f'/{LOG_FILE}'  # Provide a URL to access the log file if needed
    }), 200

def generate_frames(model):
    # Try initializing the camera
    camera = cv2.VideoCapture(0)

    # Check if the camera was opened correctly
    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        camera = None  # Set camera to None if unable to open

    # Open log file for writing defects
    with open(LOG_FILE, 'w') as log_file:
        log_file.write("Frame Number, Defect Type, Mask Area\n")  # Header for log file
        
        frame_number = 0  # Initialize frame counter

        while True:
            if camera is None:
                break  # If the camera could not be opened, break the loop
            
            success, frame = camera.read()  # Read the frame from the camera
            if not success:
                break
            
            # Run inference on the current frame
            results = model.predict(frame, stream=False)

            # Access the first result from the list
            result = results[0]

            # Extract class IDs and names from results
            class_ids = result.boxes.cls.cpu().numpy().tolist()
            class_ids = [int(cls_id) for cls_id in class_ids]
            class_names = [model.names[int(cls_id)] for cls_id in class_ids]

            mask_areas = []

            # Check if masks are available
            if result.masks is not None:
                masks = result.masks.data  # Access the raw mask data
                
                for mask in masks:
                    # Convert each mask to binary format and calculate area
                    binary_mask = mask.cpu().numpy()  # Get the numpy array representation of the mask
                    area = cv2.countNonZero(binary_mask)  # Count non-zero pixels in the mask
                    mask_areas.append(area)

                    # Log defects for each detected object (assuming one area per defect)
                    defect_type = class_names[len(mask_areas) - 1]  # Get corresponding defect type
                    
                    # Write to log file for each detected defect
                    log_file.write(f"{frame_number}, {defect_type}, {area}\n")

            else:
                print(f"No masks found for frame {frame_number}.")  # Optional logging for debugging

            # Plot the predictions on the frame
            annotated_frame = result.plot()

            # Encode the annotated frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Error: Unable to fetch frame.")
                break
            
            frame_bytes = buffer.tobytes()  # Convert to bytes

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # Yield the frame
            
            frame_number += 1  # Increment frame counter

@app.route('/video_feed')
def video_feed():
    model_choice = request.args.get('model')
    
    if model_choice == "model4":
        model = model4
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    return Response(generate_frames(model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
    elif filename == 'defect_log.csv':
        return send_from_directory(DOC_FOLDER, filename)
    
if __name__ == '__main__':
    app.run(debug=True)
