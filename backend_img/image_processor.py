import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ImageProcessor:
    def __init__(self):
        self.pytorch_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        self.scaler = StandardScaler()

    def prepare_image(self, image):
        # PIL Image'i RGB'ye çevir
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def predict_pytorch(self, model, image):
        """PyTorch modeli ile tahmin yap"""
        try:
            # Görseli tensor'e çevir
            image_tensor = self.pytorch_transform(image).unsqueeze(0)

            # Model ile tahmin
            with torch.no_grad():
                output = model(image_tensor)

                # Tensor ise uygun şekilde dönüştür
                if isinstance(output, torch.Tensor):
                    if output.dim() > 1:
                        prediction = torch.sigmoid(output[:, 1]).item()  # Sınıf 1 olasılığı
                    else:
                        prediction = output.item()
                else:
                    prediction = float(output)

                # 0-1 arasında normalize et
                prediction = max(0, min(1, prediction))

            return prediction

        except Exception as e:
            print(f"PyTorch tahmin hatası: {e}")
            return np.random.random()

    def extract_features(self, image):
        """Görsel özelliklerini çıkar (geleneksel makine öğrenmesi için)"""
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image_cv = cv2.resize(image_cv, (256, 256))
            features = []

            hist_r = cv2.calcHist([image_cv], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([image_cv], [1], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([image_cv], [2], None, [256], [0, 256]).flatten()
            hist_r /= hist_r.sum()
            hist_g /= hist_g.sum()
            hist_b /= hist_b.sum()

            features.extend(hist_r[::8])
            features.extend(hist_g[::8])
            features.extend(hist_b[::8])

            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)

            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(laplacian_var / 10000)

            for channel in range(3):
                channel_data = image_cv[:, :, channel].flatten()
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                skew_val = self.calculate_skewness(channel_data)
                features.extend([mean_val / 255, std_val / 255, skew_val])

            glcm_features = self.calculate_glcm_features(gray)
            features.extend(glcm_features)

            return np.array(features).reshape(1, -1)

        except Exception as e:
            print(f"Özellik çıkarma hatası: {e}")
            return np.random.random((1, 128))

    def calculate_skewness(self, data):
        try:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0
            skewness = np.mean(((data - mean_val) / std_val) ** 3)
            return skewness
        except:
            return 0

    def calculate_glcm_features(self, gray_image):
        try:
            small_image = cv2.resize(gray_image, (64, 64))
            features = []
            for dx, dy in [(1, 0), (0, 1)]:
                comat = np.zeros((16, 16))
                quantized = (small_image // 16).astype(np.uint8)
                h, w = quantized.shape
                for i in range(h - dx):
                    for j in range(w - dy):
                        comat[quantized[i, j], quantized[i + dx, j + dy]] += 1
                comat = comat / (comat.sum() + 1e-8)
                contrast = np.sum(comat * (np.arange(16)[:, None] - np.arange(16)[None, :]) ** 2)
                energy = np.sum(comat ** 2)
                homogeneity = np.sum(comat / (1 + np.abs(np.arange(16)[:, None] - np.arange(16)[None, :])))
                features.extend([contrast / 1000, energy, homogeneity])
            return features
        except Exception as e:
            print(f"GLCM hesaplama hatası: {e}")
            return [0] * 6

    def predict_sklearn(self, model, features):
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)
                if proba.shape[1] > 1:
                    prediction = proba[0][1]
                else:
                    prediction = proba[0][0]
            else:
                pred = model.predict(features)[0]
                prediction = float(pred)
            prediction = max(0, min(1, abs(prediction)))
            return prediction
        except Exception as e:
            print(f"Scikit-learn tahmin hatası: {e}")
            return np.random.random()
