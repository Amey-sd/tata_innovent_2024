from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

model1 = YOLO(r"/home/prakhar/Desktop/Tata Competition/Tata_Innovent_2024/models/Yolov8/Medium/weights/best.pt")
model2 = YOLO(r"/home/prakhar/Desktop/Tata Competition/Tata_Innovent_2024/models/Yolov8/Mini/weights/best.pt")
model3 = YOLO(r"/home/prakhar/Desktop/Tata Competition/Tata_Innovent_2024/models/Yolov8/Large/weights/best.pt")
model4 = YOLO(r"/home/prakhar/Desktop/Tata Competition/Tata_Innovent_2024/models/Yolov8/Large XL/weights/best.pt")
model5 = YOLO(r"/home/prakhar/Desktop/Tata Competition/Tata_Innovent_2024/models/Yolov8/Large XL/weights/best.pt")
model6 = YOLO(r"/home/prakhar/Desktop/Tata Competition/Tata_Innovent_2024/models/Yolov8/Large XL/weights/best.pt")


"""
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, r"D:/HaxS/Dataset/Car dentss.v1i.coco-segmentation/train/_annotations.coco.json", r"D:/HaxS/Dataset/Car dentss.v1i.coco-segmentation/train")
register_coco_instances("my_dataset_val", {}, r"D:/HaxS/Dataset/Car dentss.v1i.coco-segmentation/valid/_annotations.coco.json", r"D:/HaxS/Dataset/Car dentss.v1i.coco-segmentation/valid")

train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")
val_metadata = MetadataCatalog.get("my_dataset_val")
val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.OUTPUT_DIR = "D:\\HaxS\\Tata_Innovent_2024\\models\\Detectron2\\Medium"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # We have 4 classes.

cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 # Custom
predictor = DefaultPredictor(cfg)

cfg1.merge_from_file(r"D:\\HaxS\\Tata_Innovent_2024\\models\\Detectron2\\Mini\\config.yaml")  # Path to your .yaml config file
cfg1.MODEL.WEIGHTS = ("D:\\HaxS\\Tata_Innovent_2024\\models\\Detectron2\\Mini\\model_final.pth")  # Path to your fine-tuned model weights
cfg1.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Set this to the number of classes you have
model6 = DefaultTrainer(cfg1)
model6.resume_or_load(resume=False)  # Load the model weights
"""
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
    elif model_choice == "model5":
        model = model5
    elif model_choice == "model6":
        model = model6
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

    return jsonify({
        'aiResponse': 'Prediction successful',
        'image_url': f'/static/processed/{filename}',
        'helpline': 'Contact us at 1-800-CAR-HELP'
    })


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
