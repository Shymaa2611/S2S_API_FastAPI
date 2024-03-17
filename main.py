from fastapi import FastAPI, UploadFile, File
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from s2smodels import Base, Audio_segment, AudioGeneration
from pydub import AudioSegment
import os
import torch
from fastapi.responses import JSONResponse
from utils.prompt_making import make_prompt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import MarianTokenizer, MarianMTModel
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import shutil
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from io import BytesIO
import soundfile as sf
DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

app = FastAPI()

Base.metadata.create_all(engine)


@app.get("/")
def root():
    return {"message": "No result"}

#@app.post("/create_segment/")
def create_segment(start_time: float, end_time: float, audio: AudioSegment, type: str):
    session = Session()
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format='wav')
    audio_bytes = audio_bytes.getvalue()
    segment = Audio_segment(start_time=start_time, end_time=end_time, type=type, audio=audio_bytes)
    session.add(segment)
    session.commit()
    session.close()

    return {"status_code": 200, "message": "success"}


#@app.post("/generate_target/")
def generate_target(audio: AudioSegment):
    session = Session()
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format='wav')
    audio_bytes = audio_bytes.getvalue()
    target_audio = AudioGeneration(audio=audio_bytes)
    session.add(target_audio)
    session.commit()
    session.close()

    return {"status_code": 200, "message": "success"}


"""
audio segmentation into speech and non-speech using pyannote segmentation model
"""
def audio_speech_nonspeech_detection(audio_url):
    model = Model.from_pretrained(
     "pyannote/segmentation-3.0", 
      use_auth_token="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")
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
      "min_duration_on": 0.0,
      "min_duration_off": 0.0
     }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_url)
    speaker_regions=[]
    for turn, _,speaker in vad.itertracks(yield_label=True):
         speaker_regions.append({"start":turn.start,"end":turn.end})
    sound = AudioSegment.from_wav(audio_url)
    speaker_regions.sort(key=lambda x: x['start'])
    non_speech_regions = []
    for i in range(1, len(speaker_regions)):
        start = speaker_regions[i-1]['end'] 
        end = speaker_regions[i]['start']   
        if end > start:
            non_speech_regions.append({'start': start, 'end': end})
    first_speech_start = speaker_regions[0]['start']
    if first_speech_start > 0:
          non_speech_regions.insert(0, {'start': 0, 'end': first_speech_start})
    last_speech_end = speaker_regions[-1]['end']
    total_audio_duration = len(sound)  
    if last_speech_end < total_audio_duration:
            non_speech_regions.append({'start': last_speech_end, 'end': total_audio_duration})
    return speaker_regions,non_speech_regions

"""
save speech and non-speech segments in database 
"""
def split_audio_segments(audio_url):
    sound = AudioSegment.from_wav(audio_url)
    speech_segments, non_speech_segment = audio_speech_nonspeech_detection(audio_url)
    # Process speech segments
    for i, speech_segment in enumerate(speech_segments):
        start = int(speech_segment['start'] * 1000)  
        end = int(speech_segment['end'] * 1000)  
        segment = sound[start:end]
        create_segment(start_time=start/1000,
            end_time=end/1000,
            type="speech",audio=segment)
    
    # Process non-speech segments 
    for i, non_speech_segment in enumerate(non_speech_segment):
        start = int(non_speech_segment['start'] * 1000)  
        end = int(non_speech_segment['end'] * 1000)  
        segment = sound[start:end]
        create_segment(start_time=start/1000,
            end_time=end/1000,
            type="non-speech",audio=segment)

@app.post("/translate/")
def text_to_text_translation(text):
    mname = "marefa-nlp/marefa-mt-en-ar"
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = MarianMTModel.from_pretrained(mname)
    translated_tokens = model.generate(**tokenizer.prepare_seq2seq_batch([text], return_tensors="pt"))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    translated_text=" ".join(translated_text)
    return translated_text

def make_prompt_audio(name,audio_path):
    make_prompt(name=name, audio_prompt_path=audio_path)

# whisper model for speech to text process
@app.post("/speech_to_text/")   
def speech_to_text_process(segment):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
           model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
     "automatic-speech-recognition",
      model=model,
      tokenizer=processor.tokenizer,
      feature_extractor=processor.feature_extractor,
      max_new_tokens=128,
      chunk_length_s=30,
      batch_size=16,
      return_timestamps=True,
      torch_dtype=torch_dtype,
      device=device,
    )
    result = pipe(segment)
    return result["text"]

#text to speech using VALL-E-X model 
def text_to_speech(segment_id, target_text, audio_prompt):
    preload_models()
    session = Session()
    segment = session.query(Audio_segment).get(id=segment_id)
    make_prompt_audio(name=segment_id,audio_path=audio_prompt)
    audio_array = generate_audio(target_text,segment_id)
    temp_file = f"new_audio_{segment_id}.wav"
    sf.write(temp_file, audio_array, SAMPLE_RATE)  
    with open(temp_file, "rb") as f:
        segment.audio.save(f.name, File(f), save=False)
        session.commit()
        session.close()
    os.remove(temp_file)


def construct_audio():
    session = Session()
    #should be ordered by start_time
    segments = session.query(Audio_segment).order_by('start_time')
    audio_files = [AudioSegment.from_file(segment.audio.path) for segment in segments]
    target_audio = sum(audio_files,AudioSegment.empty())
    target_audio_path = "target_audio.wav"
    target_audio.export(target_audio_path, format="wav")
    generate_target(audio=target_audio_path)
    #delete all record in Audio_segment table

"""
source  => english speech
target  => arabic speeech
"""

audio_url="C:\\Users\\dell\\Downloads\\Music\\audio.wav"
@app.post("/speech2speech/")
def speech_to_speech_translation_en_ar(audio_url):
    session=Session()
    split_audio_segments(audio_url)
    speech_segments = session.query(Audio_segment).filter(Audio_segment.type == "speech").all()
    for segment in speech_segments:
        audio_data = segment.audio
        text = speech_to_text_process(audio_data)
        target_text=text_to_text_translation(text)
        segment_id = segment.id
        audio_file_path = segment.audio.path
        text_to_speech(segment_id,target_text,audio_file_path)
    construct_audio()
    return JSONResponse(status_code=200, content={"status_code": 200})



@app.get("/generate_audio/")
def get_audio(audio_url):
    speech_to_speech_translation_en_ar(audio_url)
    session=Session()
    target_audio=session.query(AudioGeneration).first()
    return target_audio
   