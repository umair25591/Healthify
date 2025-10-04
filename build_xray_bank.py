import os
import pickle
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load feature extractor
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_feature(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = model.predict(x)
    return feat[0]

folder = 'valid_xray_samples'
feature_bank = {}

for fname in os.listdir(folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(folder, fname)
        print(f"Extracting feature for {fname}...")
        feature_bank[fname] = extract_feature(path)

# Save to model/xray_feature_bank.pkl
os.makedirs('model', exist_ok=True)
with open('model/xray_feature_bank.pkl', 'wb') as f:
    pickle.dump(feature_bank, f)

print("✅ X-ray feature bank created at model/xray_feature_bank.pkl")
