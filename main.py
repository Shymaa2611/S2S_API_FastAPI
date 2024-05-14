from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from s2smodels import Base, Audio_segment, AudioGeneration
from pydub import AudioSegment
import os
from fastapi import FastAPI, Response
import torch
from fastapi.responses import JSONResponse
from utils.prompt_making import make_prompt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from io import BytesIO
from pyannote.audio import Pipeline
import soundfile as sf
from fastapi_cors import CORS
import torchaudio
from functools import lru_cache
DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

app = FastAPI()
""" 
origins = ["*"]

app.add_middleware(
    CORS,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
Base.metadata.create_all(engine)

class AudioUtility:

    def create_segment(self,start_time: float, end_time: float, audio: AudioSegment, type: str):
      session=Session()
      audio_bytes = BytesIO()
      audio.export(audio_bytes, format='wav')
      audio_bytes = audio_bytes.getvalue()
      segment = Audio_segment(start_time=start_time, end_time=end_time, type=type, audio=audio_bytes)
      session.add(segment)
      session.commit()
      session.close()
      return {"status_code": 200, "message": "success"}

    def generate_target(self,audio: AudioSegment):
      session=Session()
      audio_bytes = BytesIO()
      audio.export(audio_bytes, format='wav')
      audio_bytes = audio_bytes.getvalue()
      target_audio = AudioGeneration(audio=audio_bytes)
      session.add(target_audio)
      session.commit()
      session.close()

      return {"status_code": 200, "message": "success"}
    
    def construct_audio(self):
      session=Session()
    # Should be ordered by start_time
      segments =session.query(Audio_segment).order_by('start_time').all()
      audio_files = []
      for segment in segments:
         audio_files.append(AudioSegment.from_file(BytesIO(segment.audio), format='wav'))
      target_audio = sum(audio_files, AudioSegment.empty())
      self.generate_target(audio=target_audio)
    # Delete all records in Audio_segment table
      session.query(Audio_segment).delete()
      session.commit()
      session.close()
   
class AudioSegmentation:
    def __init__(self):
        self.audioUtility_obj=AudioUtility()
    
    @lru_cache(maxsize=None)
    def load_segmentation_model(self):
       pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
       use_auth_token="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")
       return pipeline
    
    def audio_speech_nonspeech_detection(self,audio_url):
       pipeline =self.load_segmentation_model()
       diarization = pipeline(audio_url)
       speaker_regions=[]
       for turn, _,speaker in  diarization.itertracks(yield_label=True):
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

    def split_audio_segments(self,audio_url):
       sound = AudioSegment.from_wav(audio_url)
       speech_segments, non_speech_segment = self.audio_speech_nonspeech_detection(audio_url)
       # Process speech segments
       for i, speech_segment in enumerate(speech_segments):
         start = int(speech_segment['start'] * 1000)  
         end = int(speech_segment['end'] * 1000)  
         segment = sound[start:end]
         self.audioUtility_obj.create_segment(start_time=start/1000,
            end_time=end/1000,
            type="speech",audio=segment)
       # Process non-speech segments 
       for i, non_speech_segment in enumerate(non_speech_segment):
          start = int(non_speech_segment['start'] * 1000)  
          end = int(non_speech_segment['end'] * 1000)  
          segment = sound[start:end]
          self.audioUtility_obj.create_segment(start_time=start/1000,
            end_time=end/1000,
            type="non-speech",audio=segment)

class SpeechToText:
 
   @lru_cache(maxsize=None)
   def load_whisper_model(self):
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
    return pipe
     
   def speech_to_text_process(self,segment,source_language:str):
    pipe=self.load_whisper_model()
    result = pipe(segment,generate_kwargs={"language":source_language})
    return result["text"]
   
   def speech_to_text(self,audio_url:str,source_language:str):
       pipe=self.load_whisper_model()
       result = pipe(audio_url,generate_kwargs={"language":source_language})
       return result["text"]
   
class TextTranslation:
    def detect_language(self,source_language:str,target_language:str):
       if source_language=="english":
           source_language="eng_Latn"
           if target_language=="chinese":
                target_language="zho_Hant"
           else:
               target_language="arz_Arab"
       else:
            source_language="arz_Arab"
            target_language="eng_Latn"
       return source_language,target_language

    @lru_cache(maxsize=None)
    def load_NLLB_model(self):
      pipe_trans = pipeline("translation", model="facebook/nllb-200-distilled-600M")
      return pipe_trans
    
    def text_translation(self,text,source_language:str,target_language:str):
      pipe=self.load_NLLB_model()
      source_language,target_language=self.detect_language(source_language,target_language)
      result=pipe(text,src_lang=source_language,tgt_lang=target_language)
      return result[0]['translation_text']

class TextToSpeech:
   def __init__(self):
      self.audioUtility_obj=AudioUtility()

   def make_prompt_audio(self,name,audio_path):
     make_prompt(name=name, audio_prompt_path=audio_path)
      
   def text_to_speech_process(self,segment_id, target_text, audio_prompt):
     session=Session()
     preload_models()
     segment = session.query(Audio_segment).get(segment_id)
     self.make_prompt_audio(name=f"audio_{segment_id}",audio_path=audio_prompt)
     audio_array = generate_audio(target_text,f"audio_{segment_id}")
     temp_file = BytesIO()
     sf.write(temp_file, audio_array, SAMPLE_RATE, format='wav')
     temp_file.seek(0)
     segment.audio = temp_file.getvalue()
     session.commit()
     session.close()
     temp_file.close()
    #os.remove(temp_file)

   def extract_15_seconds(self,audio_data, start_time, end_time):
    audio_segment = AudioSegment.from_file(BytesIO(audio_data), format='wav')
    start_ms = start_time * 1000  
    end_ms = min((start_time + 15) * 1000, end_time * 1000)  
    extracted_segment = audio_segment[start_ms:end_ms]
    temp_wav_path = "temp.wav"
    extracted_segment.export(temp_wav_path, format="wav")
    return temp_wav_path
   
   def check_audio_duration(self,audio_url):
    audio, sample_rate = torchaudio.load(audio_url)
    duration_in_seconds = audio.size(1) / sample_rate
    return duration_in_seconds > 15

   def split_text_into_chunks(self,text:str, chunk_size=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

   def chunk_to_speech(self, chunks):
    session = Session()
    for chunk in chunks:
        audio_array = generate_audio(chunk, prompt="person")
        temp_file_path = "temp.wav"
        sf.write(temp_file_path, audio_array, SAMPLE_RATE, format='wav')
        sound = AudioSegment.from_wav(temp_file_path)
        
        self.audioUtility_obj.create_segment(start_time=0.0,
                                             end_time=0.0,
                                             type=None,
                                             audio=sound)
        os.remove(temp_file_path)
    session.close()

   def text_to_speech(self, text: str, audio_url: str):
    chunks = self.split_text_into_chunks(text, chunk_size=10)
    preload_models()
    if self.check_audio_duration(audio_url):
        audio = AudioSegment.from_file(audio_url)
        first_15_seconds = audio[:15000]
        new_audio_path = "temp_audio.wav"
        first_15_seconds.export(new_audio_path, format="wav")
        make_prompt(name="person", audio_prompt_path=new_audio_path)
        self.chunk_to_speech(chunks)
        os.remove(new_audio_path)
    else:
        make_prompt(name="person", audio_prompt_path=audio_url)
        self.chunk_to_speech(chunks)

    self.audioUtility_obj.construct_audio()

class SpeechToSpeechTranslation:
   def __init__(self):
        self.audiosegmentation_obj=AudioSegmentation()
        self.texttranslation_obj=TextTranslation()
        self.speech2text_obj=SpeechToText()
        self.text2tspeech_obj=TextToSpeech()
        self.audioUtility_obj=AudioUtility()

   def speech_to_speech_translation(self,audio_url,source_language:str,target_language:str):
    session=Session()
    target_text=None
    self.audiosegmentation_obj.split_audio_segments(audio_url)
    #filtering by type
    speech_segments = session.query(Audio_segment).filter(Audio_segment.type == "speech").all()
    for segment in speech_segments:
        audio_data = segment.audio
        text =self.speech2text_obj.speech_to_text_process(audio_data,source_language)
        if text:
            target_text=self.texttranslation_obj.text_translation(text,source_language,target_language)
        else:
            print("speech_to_text_process function not return result. ")
        if target_text is None:
            print("Target text is None.")
        else:
           segment_id = segment.id
           segment_duration = segment.end_time - segment.start_time
           if segment_duration <=15:
              self.text2tspeech_obj.text_to_speech_process(segment_id,target_text,segment.audio)
           else:
                audio_data= self.text2tspeech_obj.extract_15_seconds(segment.audio,segment.start_time,segment.end_time)
                self.text2tspeech_obj.text_to_speech_process(segment_id,target_text,audio_data)
                os.remove(audio_data)
    self.audioUtility_obj.construct_audio()

class SpeechToTextTranslation:
   "speech to text + text2text translation"
   def __init__(self):
    self.speech2text_obj=SpeechToText()
    self.texttranslation_obj=TextTranslation()

   def speech_to_text_translation(self,audio_url:str,source_language:str,target_language:str):
      text=self.speech2text_obj.speech_to_text(audio_url,source_language)
      translated_text=self.texttranslation_obj.text_translation(text,source_language,target_language)
      return translated_text
       


@app.get("/")
async def root():
    return {"message": "No result"}

@app.get("/get_audio/")
async def get_audio(audio_url,source_language,target_language):
    speech2speechtranslation_obj=SpeechToSpeechTranslation()
    speech2speechtranslation_obj.speech_to_speech_translation(audio_url,source_language,target_language)
    session = Session()
    # Get target audio from AudioGeneration
    target_audio = session.query(AudioGeneration).order_by(AudioGeneration.id.desc()).first()
    # Remove target audio from database
    #session.delete(target_audio)
    #session.commit()
    if target_audio is None:
        raise ValueError("No audio found in the database")
    
    audio_bytes = target_audio.audio
    return Response(content=audio_bytes, media_type="audio/wav")

@app.get("/audio_segments/")
def get_all_audio_segments():
        session=Session()
        segments = session.query(Audio_segment).all()
        segment_dicts = []
        for segment in segments:
            if segment.audio is None:
                raise ValueError("No audio found in the database")

            audio_bytes = segment.audio
            file_path = f"segments//segment{segment.id}_audio.wav"
            with open(file_path, "wb") as file:
               file.write(audio_bytes)
            segment_dicts.append({
                "id": segment.id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "type": segment.type,
                "audio_url":file_path
            })
        session.close()
        return {"segments":segment_dicts}

@app.get('/get_translated_text/')
async def speech_to_text_translation(audio_url:str,source_language:str,target_language:str):
    speech2texttranslation_obj=SpeechToTextTranslation()
    translated_text=speech2texttranslation_obj.speech_to_text_translation(audio_url,source_language,target_language)
    if translated_text is None:
        return JSONResponse(status_code=404, content={"message": "No text could be extracted from the audio."})
    return JSONResponse(status_code=200, content={"translated text": translated_text})

@app.get("/get_text/")
async def speech_to_text(audio_url:str,source_language:str):
    speech2text_obj=SpeechToText()
    result=speech2text_obj.speech_to_text(audio_url,source_language)
    if result is None:
        return JSONResponse(status_code=404, content={"message": "No text could be extracted from the audio."})
    return JSONResponse(status_code=200, content={"text": result})

@app.get("/get_speech/")
async def text_to_speech(text:str,audio_url:str):
    text2speech_obj=TextToSpeech()
    text2speech_obj.text_to_speech(text,audio_url)
    session = Session()
    # Get target audio from AudioGeneration
    target_audio = session.query(AudioGeneration).order_by(AudioGeneration.id.desc()).first()
    # Remove target audio from database
    #session.delete(target_audio)
    #session.commit()
    if target_audio is None:
        raise ValueError("No audio found in the database")
    
    audio_bytes = target_audio.audio
    return Response(content=audio_bytes, media_type="audio/wav")



	