# 🖐️ Sign Language Recognition Using Hand Gestures!!!

This project uses computer vision and machine learning techniques to recognize **sign language gestures** made by hand through a webcam.

---

## 📂 Project Overview

The system captures hand gesture images, trains a model, and then detects and recognizes gestures in real time.  
It helps bridge communication between hearing-impaired individuals and others.

---

🔧 Installation & Setup (With Versions)
✅ Prerequisites

OS: Windows 10 / 11

Python Launcher: py (Python 3.x installed)

📌 Step 1: Clone the Repository
git clone https://github.com/your-username/sign-language-recognition-new.git
cd sign-language-recognition-new

📌 Step 2: Create & Activate Virtual Environment
py -m venv venv
venv\Scripts\activate

📌 Step 3: Install Required Libraries (Version Locked)
pip install flask==2.3.3
pip install numpy==1.23.5
pip install scikit-learn==1.3.2
pip install opencv-python==4.8.0.76
pip install mediapipe==0.10.9
pip install matplotlib==3.7.3


📌 Important:
These versions are required to avoid pickle compatibility issues with the trained ML model.

📌 Step 4: Run the Application
py app.py

Open in browser:

http://127.0.0.1:5000/

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
