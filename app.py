import subprocess
import sys
import pathlib

# PosixPath problemi çözümü
class PosixPath(str):
    def __new__(cls, *args, **kwargs):
        return str.__new__(cls, *args)

sys.modules['pathlib'].PosixPath = PosixPath

# Eksik paketleri yükle
try:
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.2.0"])
    import torch

try:
    import torchvision
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
    import torchvision

try:
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import hf_hub_download
from io import BytesIO
import json

from model_loader import models
from image_processor import ImageProcessor
from fact_checker import get_explanation

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return send_from_directory('pages', 'index.html')

@app.route('/<path:path>')
def serve_page(path):
    return send_from_directory('pages', path)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

# Görsel analiz
image_processor = ImageProcessor(models=models)

# Text modeli
tokenizer = DistilBertTokenizer.from_pretrained("sezerokangolge/textModels")
text_model = DistilBertForSequenceClassification.from_pretrained("sezerokangolge/textModels")

@app.route('/detect', methods=['POST'])
def detect_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Görüntü dosyası eksik'}), 400
        image_file = request.files['image']
        image = Image.open(image_file.stream)
        result = image_processor.analyze_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        input_text = data.get('text', '')

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = text_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        label = "Sahte" if predicted_class.item() == 1 else "Gerçek"
        explanation = get_explanation(input_text)

        return jsonify({
            "result": label,
            "confidence": float(confidence),
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
