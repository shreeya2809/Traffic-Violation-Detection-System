import os
import sys
import torch
import cv2
import numpy as np
import easyocr

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

# Fix Windows pathing issues
import pathlib
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Setup YOLOv5 path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DIR = os.path.join(BASE_DIR, 'yolov5')
sys.path.append(YOLO_DIR)

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator

# Device
device = select_device('cpu')

# Load all models
helmet_model = DetectMultiBackend(weights=os.path.join(BASE_DIR, 'app1', 'yolo_models', 'helmet_model.pt'), device=device)
triple_model = DetectMultiBackend(weights=os.path.join(BASE_DIR, 'app1', 'yolo_models', 'triple_seat_model.pt'), device=device)
plate_model  = DetectMultiBackend(weights=os.path.join(BASE_DIR, 'app1', 'yolo_models', 'number_plate_model.pt'), device=device)

# OCR reader
reader = easyocr.Reader(['en'])

def run_yolo_model(model, img0):
    img = cv2.resize(img0, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)

    pred = model(img)
    pred = non_max_suppression(pred)[0]

    results = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()
        for *xyxy, conf, cls in pred:
            results.append((xyxy, conf, int(cls), model.names[int(cls)]))
    return results

def detect_violations(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        image_path = fs.path(filename)

        frame = cv2.imread(image_path)
        annotator = Annotator(frame.copy(), line_width=2)

        # Helmet Detection
        helmet_detections = run_yolo_model(helmet_model, frame)
        for (xyxy, conf, cls, label) in helmet_detections:
            annotator.box_label(xyxy, f'Helmet: {label}', color=(0, 255, 0))

        # Triple Seat Detection
        triple_detections = run_yolo_model(triple_model, frame)
        for (xyxy, conf, cls, label) in triple_detections:
            annotator.box_label(xyxy, f'Triple: {label}', color=(255, 255, 0))

        # Number Plate Detection + OCR
        plate_detections = run_yolo_model(plate_model, frame)
        plate_texts = []
        for (xyxy, conf, cls, label) in plate_detections:
            x1, y1, x2, y2 = map(int, xyxy)
            plate_crop = frame[y1:y2, x1:x2]
            ocr_result = reader.readtext(plate_crop)
            plate_text = ocr_result[0][-2] if ocr_result else "Unreadable"
            plate_texts.append(plate_text)
            annotator.box_label(xyxy, f'Plate: {plate_text}', color=(0, 0, 255))

        result_img = annotator.result()
        result_path = os.path.join(settings.MEDIA_ROOT, 'result.jpg')
        cv2.imwrite(result_path, result_img)

        return render(request, 'result.html', {
            'result_image': 'result.jpg',
            'helmet_count': len(helmet_detections),
            'triple_count': len(triple_detections),
            'plates': plate_texts
        })

    return render(request, 'detect.html')
