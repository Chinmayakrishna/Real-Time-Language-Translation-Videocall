from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
import threading
import time

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/listen', methods=['POST'])
def listen():
    language_code = request.form['language']
    spoken_text, error_listen = listen_for_speech(language_code)
    
    if spoken_text:
        translated_text, error_translate = translate_text(spoken_text, language_code)
        if translated_text:
            text_to_speech(translated_text)
            return jsonify({'spoken_text': spoken_text, 'translated_text': translated_text, 'error': None})
        else:
            return jsonify({'error': error_translate})
    else:
        return jsonify({'error': error_listen})

if __name__ == '__main__':
    app.run(debug=True)
