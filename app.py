from flask import Flask, render_template, request, redirect, session
import os
import cv2
import numpy as np
import joblib
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from lsb_decoder import extract_lsb_message

app = Flask(__name__)
app.secret_key = "secret123"

# ================================
# FOLDERS
# ================================
UPLOAD_FOLDER = "static/uploads"
STEGO_FOLDER = os.path.join(UPLOAD_FOLDER, "stego")
CLEAN_FOLDER = os.path.join(UPLOAD_FOLDER, "clean")

os.makedirs(STEGO_FOLDER, exist_ok=True)
os.makedirs(CLEAN_FOLDER, exist_ok=True)

# ================================
# LOAD MODEL
# ================================
model = joblib.load("model/stego_model.pkl")

# ================================
# USER STORAGE
# ================================
users = {}

# ================================
# SCAN HISTORY STORAGE (in-memory)
# ================================
scan_history = []

# ================================
# FEATURE EXTRACTION
# ================================
def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    edges = cv2.Canny(gray, 100, 200).flatten()

    features = np.concatenate((hist, edges))
    return features.reshape(1, -1)

# ================================
# HOME
# ================================
@app.route("/")
def home():
    return render_template("home.html")

# ================================
# REGISTER
# ================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        users[username] = password
        return redirect("/login")

    return render_template("register.html")

# ================================
# LOGIN
# ================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and check_password_hash(users[username], password):
            session["user"] = username
            return redirect("/dashboard")

    return render_template("login.html")

# ================================
# LOGOUT
# ================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ================================
# DASHBOARD
# ================================
@app.route("/dashboard")
def dashboard():
    stego = len(os.listdir(STEGO_FOLDER))
    clean = len(os.listdir(CLEAN_FOLDER))
    total = stego + clean

    return render_template(
        "dashboard.html",
        stego=stego,
        clean=clean,
        total=total
    )

# ================================
# UPLOAD PAGE
# ================================
@app.route("/upload")
def upload():
    if "user" not in session:
        return redirect("/login")

    return render_template("upload.html")

# ================================
# ANALYZE IMAGE
# ================================
@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    # ================================
    # MODEL PREDICTION
    # ================================
    features = extract_features(filepath)

    if features is None:
        return "Error processing image"

    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0]) * 100

    # ================================
    # SAVE BASED ON RESULT & BUILD IMAGE URL
    # ================================
    if prediction == 1:
        save_path = os.path.join(STEGO_FOLDER, filename)
        result = "Stego"
        # Relative path from static/ for url_for
        image_static_path = "uploads/stego/" + filename
    else:
        save_path = os.path.join(CLEAN_FOLDER, filename)
        result = "Clean"
        image_static_path = "uploads/clean/" + filename

    os.replace(filepath, save_path)

    # ================================
    # EXTRACT MESSAGE
    # ================================
    try:
        message = extract_lsb_message(save_path)
        if not message or not message.strip():
            message = ""
    except Exception:
        message = ""

    # ================================
    # SAVE TO SCAN HISTORY
    # ================================
    scan_history.append({
        "filename": filename,
        "result": result,
        "confidence": round(confidence, 2),
        "image": image_static_path,
        "message": message
    })

    # ================================
    # SEND TO RESULT PAGE
    # ================================
    return render_template(
        "result.html",
        result=result,                        # FIX: was sending 'prediction' (0/1), now sends 'Clean'/'Stego'
        confidence=round(confidence, 2),
        image=image_static_path,              # FIX: now passes correct static-relative path
        message=message
    )

# ================================
# SCAN HISTORY
# ================================
@app.route("/history")
def history():
    return render_template("history.html", scans=scan_history)

# ================================
# TEST ROUTE
# ================================
@app.route("/test")
def test():
    return "Server working!"

# ================================
# RUN
# ================================
if __name__ == "__main__":
    app.run(debug=True)