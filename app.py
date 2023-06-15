import streamlit as st
import time
from audio_recorder_streamlit import audio_recorder
import whisper
import numpy as np
from scipy.io.wavfile import read, write
import io
import ffmpeg
import os 
import streamlit.components.v1 as components
from io import BytesIO
from st_custom_components import st_audiorec
import openai
import requests
import librosa.display
from matplotlib import pyplot as plt



HF_API_KEY= st.secrets["HF_API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

openai.api_key = st.secrets["OPENAI_API_KEY"]


st.set_page_config(page_title="You Speak I Write",page_icon=":ghost:", layout='wide', initial_sidebar_state="collapsed")

def plot_audio_transformations(y, sr):
    cols = [1, 1, 1]

    col1, col2, col3 = st.columns(cols)
    with col1:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Mel Spectogram</h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_transformation(y, sr, "Original"))
    with col2:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_wave(y, sr))
    with col3:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Audio</h5>",
            unsafe_allow_html=True,
        )
        spacing()
        #st.audio(create_audio_player(y, sr))
        st.audio(audio_bytes, format="audio/wav")
def plot_transformation(y, sr, transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()
def plot_wave(y, sr):
    fig, ax = plt.subplots()
    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)
    return plt.gcf()
def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

@st.cache_resource
def load_model(model_type):
   return whisper.load_model(model_type)

def load_audio(file: (str, bytes), sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: (str, bytes)
        The audio file to open or bytes of audio file

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    
    if isinstance(file, bytes):
        inp = file
        file = 'pipe:'
    else:
        inp = None
    
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def whisper_result_to_srt(result):
    text = []
    for i,s in enumerate(result['segments']):
        text.append(str(i+1))

        time_start = s['start']
        hours, minutes, seconds = int(time_start/3600), (time_start/60) % 60, (time_start) % 60
        timestamp_start = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
        timestamp_start = timestamp_start.replace('.',',')     
        time_end = s['end']
        hours, minutes, seconds = int(time_end/3600), (time_end/60) % 60, (time_end) % 60
        timestamp_end = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
        timestamp_end = timestamp_end.replace('.',',')        
        text.append(timestamp_start + " --> " + timestamp_end)

        text.append(s['text'].strip() + "\n")
            
    return "\n".join(text)

# --- Initialising SessionState ---
if "load_state" not in st.session_state:
     st.session_state.load_state = False
if "intent_state" not in st.session_state:
     st.session_state.intent_state = False
if "file_state" not in st.session_state:
    st.session_state.file_state = False
if "ask_state" not in st.session_state:
    st.session_state.ask_state = False
# model = whisper.load_model("small.en")
# #st.success("Whisper Model Loaded")

model = load_model("small")
st.success("Whisper Small Model Loaded")

st.title('⾳ ➡ ✍🏻 Transcriber')

with st.expander('About this App'):
	st.markdown('''
	This Streamlit app uses the whisper API to perform automatic speech recognition together with other useful features including: 
	- `Record` user's Audio from Mic, `Detect` Language of Speech and `transcription ` 
	- `Classify` context of Speech as Good , Bad or Neutral
	- `Upload` an audio file and return a subtitle file to download 
	- `Ask ChatGPT` in mandarin using audio from mic while providing translation of question
    


	Libraries used:
	- `streamlit` - web framework for python scripts
	- `whisper` - OpenAI's whisper library providing loading models and inference
	- `openai` - allows interaction with the ChatGPT API
	- `huggingface` - access huggingface's distilbert to perfrom intent analysis 
	- `json` - allows reading of responses from ChatGPT
	''')

# Recording
with st.container():
    st.write("---")
#--HEADER--
st.header("Recording Studio (錄音) 🎤")
# Records 3 seconds in any case
audio_bytes = audio_recorder(
  #energy_threshold=(-1.0, 1.0),
  pause_threshold=3.0,
  text="Please Speak",
  recording_color="#e8b62c",
  neutral_color="#6aa36f",
  icon_name="microphone",
  icon_size="2x",
  sample_rate=16000
)
if audio_bytes is not None:
    audio_array = load_audio(audio_bytes)

if st.checkbox(":blue[Show Audio]"):
    plot_audio_transformations(audio_array,16000)


if st.checkbox(":violet[Detect Language]"):
        audio_array = whisper.pad_or_trim(audio_array)
        mel = whisper.log_mel_spectrogram(audio_array).to(model.device)
        _, probs = model.detect_language(mel)
        st.subheader(f"Detected language: {max(probs, key=probs.get)}") 

if st.button(":green[Transcribe Audio]")or st.session_state.load_state:
        st.session_state.load_state = True
        if audio_array is not None:  
         with st.spinner(text='In progress'):
            st.success("Transcribing Audio")
            transcription = model.transcribe(audio_array)
            st.markdown(transcription["text"])
            st.success("Transcription Complete")
        
if st.button(":green[Good 👏🏻,Bad 👎🏻 or Whatever!🤷]") or st.session_state.intent_state:
        st.session_state.intent_state = True
        with st.spinner("Just a moment"):
            st.success("An AI GURU Has Answered!")
            intent_inputs = transcription["text"]
            intent_labels = ['positive','negative','neutral']
            #st.text(intent_labels)
            payload = {
                "inputs": intent_inputs,
                "parameters": {"candidate_labels": intent_labels}
                        }
            intent_output = query(payload)
            st.write(intent_output)

if not os.path.exists('data'):
    os.makedirs('data')
with st.container():
    st.write("---")
    st.header("Subtitle(字幕) Generator 📝 ")
    audio_file = st.file_uploader("Upload an audio file", type=["wav","mp3","m4a","mp4","m4p"])
    if audio_file is not None:
        st.session_state.load_state = False
        st.session_state.intent_state = False
        if st.button(":green[Transcribe the audio file]") or st.session_state.file_state:
            st.session_state.load_state = False
            st.session_state.intent_state = False
            with st.spinner("Just a moment"):
                st.success("Working hard to transcribe")
                file_transcription = model.transcribe(os.path.join('data',audio_file.name))
                st.success("Transcription Complete")
                #st.markdown(file_transcription)
                file_transcription_srt = whisper_result_to_srt(file_transcription)
                st.markdown(file_transcription_srt)
                #st.text_area(label="",value=file_transcription["text"])
                #st.markdown(file_transcription["text"])
                st.success('Done')
        # Download data
                st.download_button(":red[Download]", data=file_transcription_srt, file_name="{}.{}".format(audio_file.name,"srt"))
with st.container():
    st.write("---")
    st.header("Speak Directly to ChatGPT 🤖 不再打字 ⌨️")
    
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
    # # display audio data as received on the backend
    #     st.audio(wav_audio_data, format='audio/wav')
    
# INFO: by calling the function an instance of the audio recorder is created
# INFO: once a recording is completed, audio data will be saved to wav_audio_data
        st.session_state.load_state = False
        st.session_state.intent_state = False
        with st.spinner("Processing Audio"):
            chat_transcription = model.transcribe(load_audio(wav_audio_data),fp16=False)
            chat_translations = model.transcribe(load_audio(wav_audio_data),task='translate') 
            chat_question = chat_transcription["text"]
        st.session_state.ask_state = False
        col1, col2= st.columns(2)
        with col1:
            st.text_area(label="Question from user", value=[chat_question,chat_translations["text"]])
        with col2:
            #if st.button(":orange[Ask GPT]"):
                with st.spinner("Just a moment"):
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                                {"role": "system", "content": "You are a chatbot"},
                                {"role": "user", "content": f"{chat_question}"},
                            ]
                    )
                    chat_answer = ''
                    for choice in response.choices:
                        chat_answer += choice.message.content
                        st.text_area(label="Answer from ChatGPT", value=chat_answer,height=500)



# API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
# #headers = {"Authorization": "Bearer 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"}


# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()

# output = query({
#     "inputs": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
#     "parameters": {"candidate_labels": ["refund", "legal", "faq"]},
# })
# output