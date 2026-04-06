# Face Recognition-Based Access System - Set Up GUI

## 📌 Project Overview

This project is a **Face Recognition-Based Access Control System** built using Python. It uses a graphical user interface (GUI) to allow users to:

* Register faces (via camera or uploaded images)
* Train a machine learning model
* Recognize faces in real-time
* Log access attempts (granted/denied)
* View reports and system performance

This system is useful for:

* Security systems
* Attendance tracking
* Smart access control

---

## 🧠 How the System Works (Simple Explanation)

1. **Face Capture**

   * The system captures images of a person’s face.

2. **Face Alignment**

   * Faces are aligned using MediaPipe to improve accuracy.

3. **Feature Extraction**

   * A deep learning model (MobileNetV2) extracts features from faces.

4. **Classification**

   * An SVM (Support Vector Machine) classifier learns to recognize different people.

5. **Recognition**

   * When a face is detected, it is compared with trained data and classified.

---

## 🛠️ Requirements

### 📦 Software Requirements

Make sure you have:

* Python 3.8+
* pip (Python package manager)

### 📚 Install Dependencies

Run this in your terminal:

```bash
pip install opencv-python numpy tensorflow scikit-learn mediapipe pillow pandas pygame joblib albumentations matplotlib seaborn pillow-heif
```

---

## 📁 Project Structure

```
project_folder/
│
├── face_access_control_gui.py   # Main application file
├── dataset/                     # Stores captured face images
├── model/                       # Stores trained models
├── granted_photos/              # Saved images for granted access
├── denied_photos/               # Saved images for denied access
├── Training_Report/             # Training reports and graphs
├── access_log.csv               # Logs of system activity
└── haarcascade_frontalface_default.xml  # Face detection file
```

---

## 🚀 How to Run the Project

1. Open terminal in your project folder
2. Run:

```bash
python face_access_control_gui.py
```

3. The GUI window will open

---

## 🧑‍💻 Features and How to Use Them

### 1. 👤 Register User

#### Option A: Capture from Camera

* Click **Register User → Capture From Camera**
* Enter a name (only letters/numbers)
* The system will:

  * Detect your face
  * Capture multiple images
  * Save them in the dataset folder

#### Option B: Upload Images

* Click **Register User → Upload Image File**
* Select images from your computer
* The system will:

  * Detect faces
  * Apply augmentation (rotation, brightness, etc.)
  * Save processed images

---

### 2. 🔁 Retrain Model

* Click **Retrain Model**
* The system will:

  * Load all images
  * Apply augmentations (blur, fog, rain, etc.)
  * Train the deep learning model
  * Train the SVM classifier
  * Generate reports

⚠️ Important:

* You need **at least 2 users** to train the model

---

### 3. 🎥 Start Face Recognition

* Click **Start Face Recognition**
* The system will:

  * Open your webcam
  * Detect faces in real time
  * Display:

    * Name
    * Confidence score

---

### 4. 🧪 Test Model

* Click **Test Model**
* Upload test images
* The system will:

  * Predict identities
  * Generate accuracy reports
  * Save confusion matrix and graphs

---

### 5. 🗑️ Delete User

* Select a user from dropdown
* Click **Delete User**
* This removes:

  * Their dataset
  * Their contribution to the model

---

### 6. 📊 View Logs

* Logs are stored in `access_log.csv`
* Contains:

  * Timestamp
  * Name
  * Confidence
  * Status

---

### 7. 📈 Training Reports

After training, the system generates:

* Accuracy report
* Confusion matrix
* Cross-validation scores
* Performance graphs

These are saved in:

```
Training_Report/
```

---

### 8. ⚙️ Settings

You can adjust:

* Confidence threshold
* Distance threshold

This affects recognition accuracy.

---

### 9. 🌙 Dark Mode

* Toggle between light and dark themes

---

## 📊 Machine Learning Details (Beginner Friendly)

### 🔹 MobileNetV2

* A pre-trained deep learning model
* Extracts facial features (embeddings)

### 🔹 SVM (Support Vector Machine)

* Classifies faces based on extracted features

### 🔹 Data Augmentation

The system artificially increases data using:

* Blur
* Fog
* Rain
* Brightness changes
* Noise

This improves real-world performance.

---

## ⚠️ Common Issues & Fixes

### ❌ Camera not working

* Check if camera index is correct
* Ensure no other app is using the camera

---

### ❌ Model not training

* Ensure at least 2 users exist
* Check dataset folder

---

### ❌ Low accuracy

* Capture more images per user
* Ensure good lighting
* Avoid blurry images

---

## 💡 Best Practices

* Use **at least 50–100 images per user**
* Capture faces from different angles
* Use good lighting conditions
* Avoid duplicates

---

## 📌 Future Improvements

You can extend this project by:

* Adding database integration
* Deploying on a web app
* Adding liveness detection (anti-spoofing)
* Using more advanced models (e.g., FaceNet)

---

## 👨‍💻 Author

Developed as a Face Recognition Access Control System with GUI using Python and Machine Learning.

---

## 📜 License

This project is open-source. You can modify and use it for learning or projects.

---

## 🙌 Final Note

This project combines:

* Computer Vision
* Deep Learning
* GUI Development

If you're a beginner, take it step by step:

1. Understand the GUI
2. Learn how images are processed
3. Explore how the model is trained

You’ll gain strong skills in AI + Software Development 🚀
