import pytest
from fastapi.testclient import TestClient
from main import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
client = TestClient(app)

@pytest.fixture(scope="module")
def test_db():
    engine = create_engine('sqlite:///:memory:')
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    yield db
    db.close()

def test_create_segment(test_db):
    audioutility_obj=AudioUtility()
    start_time = 0.0
    end_time = 10.0
    type = "speech"
    audio = AudioSegment.silent(duration=1000)  
    response =audioutility_obj.create_segment(start_time, end_time, audio, type)
    assert response == {"status_code": 200, "message": "success"}
    created_segment = test_db.query(Audio_segment).filter_by(start_time=start_time, end_time=end_time, type=type).first()
    assert created_segment is not None
    assert created_segment.start_time 
    assert created_segment.end_time == end_time
    assert created_segment.type == type

def test_generate_target(test_db):
    audioutility_obj=AudioUtility()
    audio = AudioSegment.silent(duration=1000)  
    response =audioutility_obj.generate_target(audio)
    assert response == {"status_code": 200, "message": "success"}
    created_audio = test_db.query(AudioGeneration).order_by(AudioGeneration.id.desc()).first()
    assert created_audio is not None

def test_audio_speech_nonspeech_detection():
    audiosegment_obj=AudioSegmentation()
    audio_url="C:\\Users\\dell\\Downloads\\Music\\audio.wav"
    speech_segments, non_speech_segments =audiosegment_obj.audio_speech_nonspeech_detection(audio_url)
    assert len(speech_segments) > 0
    assert len(non_speech_segments) > 0

def test_text_translation():
    texttranslation_obj=TextTranslation()
    text="How are you today"
    source_language="english"
    target_language="arabic"
    target_text=texttranslation_obj.text_translation(text,source_language,target_language)
    assert target_text is not None

def test_speech_to_speech_translation():
    speech2speech_obj=SpeechToSpeechTranslation()
    audio_url="C:\\Users\\dell\\Downloads\\Music\\audio.wav"
    source_language="english"
    target_language="arabic"
    response = speech2speech_obj.speech_to_speech_translation(audio_url,source_language,target_language)
    assert response.status_code == 200
    assert response.json() == {"status_code":"succcessfully"}

def test_speech_to_text():
    test_params = {
        "audio_url": "C:\\Users\\dell\\Downloads\\Music\\audio.wav",
        "target_language": "arabic"
    }
    response = client.get("/get_text", params=test_params)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    data = response.json()
    assert "text" in data, "Response does not contain text"
    print("Test passed: Text retrieved successfully")

def test_speech_to_text_translation():
    
    test_params = {
        "audio_url": "C:\\Users\\dell\\Downloads\\Music\\audio.wav",
        "target_language": "arabic"
    }
    response = client.get("/get_translated_text", params=test_params)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    data = response.json()
    assert "translated text" in data, "Response does not contain translated text"
    print("Test passed:Successfully retrived translated text")

def test_text_to_speech():
    test_params = {
        "text": "Hello, this is a test.",
        "audio_url": "C:\\Users\\dell\\Downloads\\Music\\audio.wav"
    }
    response = client.get("/get_speech", params=test_params)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    assert response.headers["Content-Type"] == "audio/wav", "Response is not audio"
    print("Test passed:Audio retrieved successfully")

def test_speech_to_speech_translation():
    client = TestClient(app)
    test_params = {
        "audio_url": "C:\\Users\\dell\\Downloads\\Music\\audio.wav",
        "source_language": "english",
        "target_language": "arabic"
    }
    response = client.get("/get_audio/", params=test_params)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    assert response.headers["Content-Type"] == "audio/wav", "Response does not contain audio data"
    print("Test passed: Audio retrieved successfully")

def test_get_all_audio_segments():
    response = client.get("/audio_segments/")
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    assert "segments" in response.json(), "Response does not contain segments"
    segments = response.json()["segments"]
    for segment in segments:
        assert "id" in segment, "Segment ID not found in response"
        assert "start_time" in segment, "Start time not found in response"
        assert "end_time" in segment, "End time not found in response"
        assert "type" in segment, "Type not found in response"
        assert "audio_url" in segment, "Audio URL not found in response"
    print("Test passed: segments are retrived sucessfully")
