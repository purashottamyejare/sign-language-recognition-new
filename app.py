import os
from datetime import time
from random import random
import time

from flask import Flask, render_template, Response, jsonify, send_file, url_for
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# This list maps detected sign to alphabet
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'j', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
               17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# This list maps alphabet to object
alphabet_list = {'A': 'Apple', 'B': 'Ball', 'C': 'Cat', 'D': 'Dog', 'E': 'Elephant', 'F': 'Frog', 'G': 'Goat', 'H': 'Hen',  'I': 'Ink', 'J': 'Joker', 'K': 'Kite', 'L': 'Lion',
                 'M': 'Mangoes', 'N': 'Nose', 'O': 'Orange', 'P': 'Parrot', 'Q': 'Quack', 'R': 'Rainbow', 'S': 'Star', 'T': 'Telephone', 'U': 'Umbrella', 'V': 'Van', 'W': 'Watch',
                 'X': 'X-Ray', 'Y': 'Yak', 'Z': 'Zebras'}

# This list maps alphabet to facts
facts_list = {
    'A': 'Did you know that apples float in water because they are made up of 25% air? This is why they bob up and down when you put them in a bowl of water!',
    'B': 'Balls are round objects used in sports and games; they can be made from various materials such as rubber, leather, or plastic and are designed to bounce, roll, or fly depending on the game or sport they are used for.',
    'C': 'Cats have a special way of seeing in the dark because their eyes can reflect light like a mirror!',
    'D': 'Dogs are really good at learning tricks and commands because they love to make their owners happy!',
    'E': 'Elephants are the largest land animals on Earth, and they have big ears that they use to help them stay cool in hot weather!',
    'F': 'Some species of frogs can jump up to 20 times their own body length in a single leap!',
    'G': 'Goats are known for their incredible agility and climbing skills.',
    'H': 'Hens have a special communication skill called "tidbitting.',
    'I': 'The word "ink" itself comes from the Latin word "encaustum," which means burned in.',
    'J': 'Jokers were known for their colorful costumes, exaggerated antics, and sharp wit, much like the modern interpretation of the Joker character in comics and films.',
    'K': 'Kite invented over 2,000 years ago in China.',
    'L': 'Lions is that they are the only cats that live in groups, known as prides. ',
    'M': ' Mangoes are believed to be one of the oldest cultivated fruits',
    'N': 'Nose are as unique as fingerprints! No two people have exactly the same nose shape and structure.',
    'O': 'The word "orange" comes from the Old French "orenge," which originated from the Arabic word "naranj."',
    'P': 'Parrots are incredibly intelligent birds, known for their ability to mimic human speech and other sounds.',
    'Q': 'Quack bird, commonly known as the duck, is that their distinctive "quack" sound does not actually occur in all duck species.',
    'R': 'A fascinating fact about rainbows is that they are actually full circles, not just arcs! ',
    'S': 'A fascinating fact about stars is that they come in various sizes, colors, and temperatures.',
    'T': 'A fun fact about the telephone is that the very first words spoken on it were by its inventor, Alexander Graham Bell, to his assistant.',
    'U': 'A fun fact about umbrellas is that they were originally used not for protection against rain, but for protection against the sun.',
    'V': 'A fun fact about vans is that the term van originated from the word "caravan, which initially referred to a covered vehicle used for transporting goods or people',
    'W': 'Watches is that the world first wristwatch was created by Patek Philippe, a Swiss luxury watch manufacturer, in 1868',
    'X': 'A fascinating fact about X-rays is that they were discovered accidentally',
    'Y': 'A fun fact about yaks is that they are incredibly well adapted to high-altitude environments.',
    'Z': 'A fun fact about zebras is that their distinctive black and white stripes serve several purposes. '

}

# This variable is used to store predicted result and update it on webpage asynchronously
predicted_character = ''


def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try alternative camera index
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        # Return error frame if camera can't be opened
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera not accessible", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            global predicted_character
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    global predicted_character  # Access the global variable
    return render_template('index.html')


@app.route('/detect')  # Sign Detection page
def detect():
    global predicted_character  # Access the global variable
    return render_template('hand_sign_detect old.html')


@app.route('/video_feed')  # Video output
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detected_char')  # Display detected alphabet and object name asynchronously
def detected_char():
    global predicted_character
    display_text = ''
    if predicted_character == '':
        display_text = 'Not Detected'
    else:
        display_text = predicted_character + " for " + alphabet_list[predicted_character]
    return jsonify({'character': display_text})


@app.route('/object_image')  # Display Image related to alphabet detected
def object_image():
    global predicted_character
    if predicted_character == '':
        image_url = url_for('static', filename='logo.png')
    else:
        # Path to your images folder
        image_filename = predicted_character + '.png'
        # Create the full path to the image using url_for for proper URL generation
        image_url = url_for('static', filename=image_filename)
    return jsonify({'image_url': image_url})


@app.route('/informative_fact')  # Display informative fact about alphabet detected
def informative_fact():
    global predicted_character
    display_text = ''
    if predicted_character == '':
        display_text = 'FUN FACT HERE!!!'
    else:
        display_text = "Fun Fact about " + alphabet_list[predicted_character] + ', ' + facts_list[predicted_character]
    return jsonify({'fact': display_text})


if __name__ == '__main__':
    app.run(debug=True)
