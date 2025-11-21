import io
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pickle
import streamlit as st
from PIL import Image


@st.cache_resource
def load_model():
    model_dict = pickle.load(open('model.p', 'rb'))
    return model_dict['model']


model = load_model()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

alphabet_list = {
    'A': 'Apple', 'B': 'Ball', 'C': 'Cat', 'D': 'Dog', 'E': 'Elephant', 'F': 'Frog', 'G': 'Goat', 'H': 'Hen',
    'I': 'Ink', 'J': 'Joker', 'K': 'Kite', 'L': 'Lion', 'M': 'Mangoes', 'N': 'Nose', 'O': 'Orange', 'P': 'Parrot',
    'Q': 'Quack', 'R': 'Rainbow', 'S': 'Star', 'T': 'Telephone', 'U': 'Umbrella', 'V': 'Van', 'W': 'Watch',
    'X': 'X-Ray', 'Y': 'Yak', 'Z': 'Zebras'
}

facts_list = {
    'A': 'Did you know that apples float in water because they are made up of 25% air? '
         'This is why they bob up and down when you put them in a bowl of water!',
    'B': 'Balls are round objects used in sports and games; they can be made from various materials such as rubber, '
         'leather, or plastic and are designed to bounce, roll, or fly depending on the game or sport they are used for.',
    'C': 'Cats have a special way of seeing in the dark because their eyes can reflect light like a mirror!',
    'D': 'Dogs are really good at learning tricks and commands because they love to make their owners happy!',
    'E': 'Elephants are the largest land animals on Earth, and they have big ears that they use to help them stay cool '
         'in hot weather!',
    'F': 'Some species of frogs can jump up to 20 times their own body length in a single leap!',
    'G': 'Goats are known for their incredible agility and climbing skills.',
    'H': 'Hens have a special communication skill called "tidbitting."',
    'I': 'The word "ink" itself comes from the Latin word "encaustum," which means burned in.',
    'J': 'Jokers were known for their colorful costumes, exaggerated antics, and sharp wit, '
         'much like the modern interpretation of the Joker character in comics and films.',
    'K': 'Kite invented over 2,000 years ago in China.',
    'L': 'Lions is that they are the only cats that live in groups, known as prides.',
    'M': 'Mangoes are believed to be one of the oldest cultivated fruits.',
    'N': 'Nose are as unique as fingerprints! No two people have exactly the same nose shape and structure.',
    'O': 'The word "orange" comes from the Old French "orenge," which originated from the Arabic word "naranj."',
    'P': 'Parrots are incredibly intelligent birds, known for their ability to mimic human speech and other sounds.',
    'Q': 'Quack bird, commonly known as the duck, is that their distinctive "quack" sound does not actually occur in all duck species.',
    'R': 'A fascinating fact about rainbows is that they are actually full circles, not just arcs!',
    'S': 'A fascinating fact about stars is that they come in various sizes, colors, and temperatures.',
    'T': 'A fun fact about the telephone is that the very first words spoken on it were by its inventor, Alexander Graham Bell, to his assistant.',
    'U': 'A fun fact about umbrellas is that they were originally used not for protection against rain, but for protection against the sun.',
    'V': 'A fun fact about vans is that the term van originated from the word "caravan," which initially referred to a covered vehicle used for transporting goods or people.',
    'W': 'Watches is that the world first wristwatch was created by Patek Philippe, a Swiss luxury watch manufacturer, in 1868.',
    'X': 'A fascinating fact about X-rays is that they were discovered accidentally.',
    'Y': 'A fun fact about yaks is that they are incredibly well adapted to high-altitude environments.',
    'Z': 'A fun fact about zebras is that their distinctive black and white stripes serve several purposes.'
}


def predict_from_image(pil_image: Image.Image):
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    H, W, _ = frame.shape
    data_aux = []
    x_, y_ = [], []

    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    x1 = int(min(x_) * W) - 10
    y1 = int(min(y_) * H) - 10
    x2 = int(max(x_) * W) - 10
    y2 = int(max(y_) * H) - 10

    prediction = model.predict([np.asarray(data_aux)])
    predicted_character = labels_dict[int(prediction[0])]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(
        frame,
        predicted_character,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    return predicted_character, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_related_image(letter: str):
    static_dir = Path('static')
    for candidate in (static_dir / f'{letter}.png', static_dir / f'{letter.lower()}.png'):
        if candidate.exists():
            return Image.open(candidate)
    return Image.open(static_dir / 'logo.png')


def load_reference_image():
    return Image.open(Path('static') / 'gestures.jpg')


st.set_page_config(page_title='Hand Sign Classifier', page_icon='✋', layout='wide')

if 'show_detector' not in st.session_state:
    st.session_state.show_detector = False

if 'show_reference' not in st.session_state:
    st.session_state.show_reference = False

st.title('SignSpark Learning Lab')
st.markdown(
    'Welcome to a kid-friendly sandbox where hand signs become stories. '
    'Learn the alphabet with gestures, see colorful objects, and read curiosity-sparking facts!'
)

cta_col, reference_col, info_col = st.columns([1, 1, 2])
with cta_col:
    if st.button('👉 Check Gestures Now', use_container_width=True):
        st.session_state.show_detector = True
with reference_col:
    if st.button('📘 American Gestures Reference', use_container_width=True):
        st.session_state.show_reference = True
with info_col:
    st.info('Need a refresher first? Explore the home section, then click **Check Gestures Now** to open the camera.')

if st.session_state.show_reference:
    reference_container = st.container()
    with reference_container:
        st.image(
            load_reference_image(),
            caption='American Sign Language alphabet',
            use_container_width=True
        )
        if st.button('Close reference', key='close_reference'):
            st.session_state.show_reference = False

st.divider()

if st.session_state.show_detector:
    st.subheader('Live Gesture Checker')
    col_cam, col_result = st.columns([2, 1])

    with col_cam:
        st.caption('Show your hand sign and snap a photo')
        camera_input = st.camera_input('Camera', key='detector_camera')
        annotation_placeholder = st.empty()

    with col_result:
        st.caption('Realtime insights')
        result_placeholder = st.empty()
        object_placeholder = st.empty()
        fact_placeholder = st.empty()

    if camera_input is None:
        st.info('Waiting for camera input…')
        annotation_placeholder.empty()
        object_placeholder.empty()
        fact_placeholder.empty()
    else:
        image = Image.open(io.BytesIO(camera_input.getvalue())).convert('RGB')
        predicted_char, annotated = predict_from_image(image)

        if predicted_char is None:
            result_placeholder.warning('No hand detected. Try again with better lighting and keep your hand within the frame.')
            st.image(image, caption='Last capture')
            annotation_placeholder.empty()
            object_placeholder.empty()
            fact_placeholder.empty()
        else:
            object_name = alphabet_list[predicted_char]
            fact = facts_list[predicted_char]

            result_placeholder.success(f'Predicted: **{predicted_char}** — {object_name}')
            annotation_placeholder.image(annotated, caption='Model annotation', width=420)
            related_image = load_related_image(predicted_char)

            object_placeholder.image(related_image, caption=f'{predicted_char} for {object_name}', width=300)
            fact_placeholder.success(fact)
else:
    st.subheader('Home')
    st.write(
        'This space introduces sign-language basics using playful visuals. '
        'When you are ready to try your own gestures, hit the **Check Gestures Now** button above.'
    )

st.sidebar.title('About the Project')
st.sidebar.markdown(
    'This interactive sign-language explorer is designed for kids and educators: snap a picture of a hand sign, learn the corresponding alphabet letter, see a fun object, and read an engaging fact to reinforce learning.'
)

