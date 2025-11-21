# 🖐️ Sign Language Recognition Using Hand Gestures!!!

This project uses computer vision and machine learning techniques to recognize **sign language gestures** made by hand through a webcam.

---

## 📂 Project Overview

The system captures hand gesture images, trains a model, and then detects and recognizes gestures in real time.  
It helps bridge communication between hearing-impaired individuals and others.

---

## 🚀 How to Run the Project

### 1️⃣ Step 1 — Collect Hand Gesture Images

Run the following file:
python collect_img.py

2️⃣ Step 2 — Run the Application

After collecting images, run:
python app.py

This will start the Flask web server.

The terminal will display a URL (something like http://127.0.0.1:5000).

Open that URL in your browser to test the gesture recognition functionality.

note- first run collect_img.py and check how its working, You may need to update code to save photos, so go with script and check how many photos You want for each gesture.

Note- Ignore streamlit_app.py, it is used to deploy project to the live server
