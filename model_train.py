import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ================================
# 1. DATASET PATH
# ================================
base_path = "F:/steganography-detector/dataset/train/train"

clean_path = os.path.join(base_path, "clean")
stego_path = os.path.join(base_path, "stego")

# ================================
# 2. FEATURE EXTRACTION
#    ⚠️ Must match app.py exactly!
# ================================
def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"  ⚠️  Could not read: {image_path}")
        return None

    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist  = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    edges = cv2.Canny(gray, 100, 200).flatten()

    features = np.concatenate((hist, edges))
    return features  # shape: (16640,)

# ================================
# 3. COLLECT FILE PATHS
# ================================
def collect_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(root, f))
    return paths

clean_files = collect_images(clean_path)
stego_files = collect_images(stego_path)

print(f"📁 Clean images found : {len(clean_files)}")
print(f"📁 Stego  images found: {len(stego_files)}")

if not clean_files or not stego_files:
    raise FileNotFoundError(
        "❌ No images found. Check your dataset path:\n"
        f"   clean → {clean_path}\n"
        f"   stego → {stego_path}"
    )

# ================================
# 4. BALANCE DATASET
# ================================
min_size = min(len(clean_files), len(stego_files))
clean_files = clean_files[:min_size]
stego_files = stego_files[:min_size]
print(f"⚖️  Balanced dataset   : {min_size} samples per class ({min_size * 2} total)\n")

# ================================
# 5. LOAD FEATURES & LABELS
# ================================
data   = []
labels = []

print("🔄 Extracting features from clean images...")
for i, path in enumerate(clean_files, 1):
    features = extract_features(path)
    if features is not None:
        data.append(features)
        labels.append(0)
    if i % 500 == 0:
        print(f"   {i}/{min_size} done")

print(f"✅ Clean done: {labels.count(0)} samples\n")

print("🔄 Extracting features from stego images...")
for i, path in enumerate(stego_files, 1):
    features = extract_features(path)
    if features is not None:
        data.append(features)
        labels.append(1)
    if i % 500 == 0:
        print(f"   {i}/{min_size} done")

print(f"✅ Stego done: {labels.count(1)} samples\n")

# ================================
# 6. CONVERT TO NUMPY
# ================================
X = np.array(data)
y = np.array(labels)

print(f"📊 Total samples : {len(X)}")
print(f"📐 Feature shape : {X.shape}\n")

# Sanity check — must match app.py's extract_features output
assert X.shape[1] == 16640, (
    f"❌ Feature size mismatch! Got {X.shape[1]}, expected 16640.\n"
    "   Make sure model_train.py and app.py use the same extract_features()."
)

# ================================
# 7. TRAIN / TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Train samples : {len(X_train)}")
print(f"📊 Test  samples : {len(X_test)}\n")

# ================================
# 8. TRAIN MODEL
# ================================
print("🚀 Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,       # more trees = better accuracy
    max_depth=None,         # let trees grow fully
    min_samples_split=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1               # use all CPU cores → faster
)

model.fit(X_train, y_train)
print("✅ Training complete!\n")

# ================================
# 9. EVALUATE
# ================================
predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")
print("📊 Classification Report:")
print(classification_report(y_test, predictions, target_names=["Clean", "Stego"]))

# ================================
# 10. SAVE MODEL
# ================================
os.makedirs("model", exist_ok=True)
model_path = "model/stego_model.pkl"
joblib.dump(model, model_path)

print(f"💾 Model saved → {model_path}")
print("🎉 Done!")