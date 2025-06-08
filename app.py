from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import hf_hub_download
from PIL import Image
from io import BytesIO
import base64
import json
import subprocess
import sys

try:
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.2.0"])
    import torch
    
# Görsel analiz için
from model_loader import models
from image_processor import ImageProcessor

# Metin açıklaması için
from fact_checker import get_explanation



# Flask uygulaması
app = Flask(__name__)
CORS(app)

# Ana sayfa statik sayfaları
@app.route('/')
def home():
    return send_from_directory('pages', 'index.html')

@app.route('/<path:path>')
def serve_page(path):
    return send_from_directory('pages', path)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

# Görsel modelleri yükle
image_processor = ImageProcessor(models=models)

# HuggingFace text modeli (DistilBERT) yükle
tokenizer = DistilBertTokenizer.from_pretrained("sezerokangolge/textModels")
text_model = DistilBertForSequenceClassification.from_pretrained("sezerokangolge/textModels")

# ✅ GÖRSEL ANALİZ ENDPOINT
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

# ✅ METİN ANALİZİ ENDPOINT
@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        input_text = data.get('text', '')

        # Model tahmini
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = text_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

        # Tahmin sonucu
        label = "Sahte" if predicted_class.item() == 1 else "Gerçek"

        # Açıklama
        explanation = get_explanation(input_text)

        return jsonify({
            "result": label,
            "confidence": float(confidence),
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Sunucu başlatma
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
