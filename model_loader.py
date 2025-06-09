import torch
import pickle
import sys
import os
import pathlib
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download

HF_REPO = "sezerokangolge/imgModels"

class PosixPath(str):
    def __new__(cls, *args, **kwargs):
        return str.__new__(cls, *args)

sys.modules['pathlib'].PosixPath = PosixPath

def get_my_x(o):
    return o

def get_my_y(o):
    return o

sys.modules['__main__'].get_my_x = get_my_x
sys.modules['__main__'].get_my_y = get_my_y

from model import (
    PatchEmbedding, Transformer, VIT,
    TransformerBlock, residual, multiHeadAttention
)

sys.modules['__main__'].PatchEmbedding = PatchEmbedding
sys.modules['__main__'].Transformer = Transformer
sys.modules['__main__'].VIT = VIT
sys.modules['__main__'].TransformerBlock = TransformerBlock
sys.modules['__main__'].residual = residual
sys.modules['__main__'].multiHeadAttention = multiHeadAttention

def load_sklearn_model(filename):
    try:
        url = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
        response = requests.get(url)
        model = pickle.load(BytesIO(response.content), encoding="latin1")
        print(f"[✓] Sklearn modeli yüklendi: {filename}")
        return model
    except Exception as e:
        print(f"[X] Sklearn model hatası: {filename} → {e}")
        return None

def load_torch_model(filename):
    try:
        model_path = hf_hub_download(repo_id=HF_REPO, filename=filename)
        model = torch.load(model_path, map_location=torch.device("cpu"))
        print(f"[✓] Torch modeli yüklendi: {filename}")
        return model
    except Exception as e:
        print(f"[X] Torch model hatası: {filename} → {e}")
        return None

model_files = [
    "Copy_Move_FIM_fixed.pkl",
    "Inpainting_FIM_fixed.pkl",
    "Splicing_FIM_fixed.pkl",
    "Deepfake_Model_1.pth",
    "Deepfake_Model_2.pth",
    "Fake_Face_Detection_Model.pth",
    "Fake_Face_Detection_Model_1.pth",
    "Fake_Face_Detection_Model_2.pth",
    "model_2.pth"
]

models = []
for filename in model_files:
    if filename.endswith(".pkl"):
        models.append(load_sklearn_model(filename))
    elif filename.endswith(".pth"):
        models.append(load_torch_model(filename))
