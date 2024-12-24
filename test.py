from flask import Flask, render_template, request, jsonify,Response
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
import threading
import time
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

app = Flask(__name__)

# Initialize recognizer, translator, and other components
translator = Translator()

def listen_for_speech(language_code):
    language_map = {
        'kn': 'kn-IN',  # Kannada
        'hi': 'hi-IN',  # Hindi
        'ml': 'ml-IN'   # Malayalam
    }
    
    language_name = {
        'kn': 'Kannada',
        'hi': 'Hindi',
        'ml': 'Malayalam'
    }
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()        
    language_code = language_code.lower()
    
    if language_code not in language_map:
        return None, f"{language_code} is not a supported language."
    
    with microphone as source:
        print(f"Listening for {language_name[language_code]} speech...")
        recognizer.pause_threshold = 0.5
        audio = recognizer.listen(source)
    try:
        spoken_text = recognizer.recognize_google(audio, language=language_map[language_code])
        print(f"You said ({language_name[language_code]}): " + spoken_text)
        return spoken_text, None
    except sr.UnknownValueError:
        return None, "Could not understand audio"
    except sr.RequestError as e:
        return None, f"Could not request results; {e}"

def translate_text(text, src_lang):
    try:
        translated = translator.translate(text, src=src_lang, dest='en')
        print("Translation: " + translated.text)
        return translated.text, None
    except Exception as e:
        return None, f"Translation error: {e}"

def text_to_speech(text):
    try:
        tts = gTTS(text, lang='en')
        tts_fp = io.BytesIO()
        tts.write_to_fp(tts_fp)
        tts_fp.seek(0)
        tts_audio = AudioSegment.from_file(tts_fp, format="mp3")
        
        # Play the audio in a non-blocking way
        play_thread = threading.Thread(target=play, args=(tts_audio,))
        play_thread.start()
    except Exception as e:
        print(f"Text-to-speech error: {e}")

# Emotion detection setup
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def listen_and_process(language_code):
    spoken_text, error_listen = listen_for_speech(language_code)
  
    if spoken_text:
        translated_text, error_translate = translate_text(spoken_text, language_code)
        print(translate_text)
        if translated_text:
            text_to_speech(translated_text)
            return spoken_text, translated_text, None
        else:
            return None, None, error_translate
    else:
        return None, None, error_listen

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/single')
def single():
    return render_template('single.html')




@app.route('/listen', methods=['POST'])
def listen():
    print("vghjkl")
    language_code = request.form['language']
    print(language_code)
    spoken_text, translated_text, error = listen_and_process(language_code)
    
    if error:
        return jsonify({'error': error})
    else:
        return jsonify({'spoken_text': spoken_text, 'translated_text': translated_text, 'error': None})

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

if __name__ == '__main__':
    app.run(debug=True)
