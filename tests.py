import pytest
from fastapi.testclient import TestClient
from main import *
from unittest.mock import Mock
from pydub import AudioSegment
client = TestClient(app)

def test_create_segment():
    audio_url='C:\\Users\\dell\\Downloads\\Music\\audio.wav'
    start=5*1000
    end=10*1000
    sound = AudioSegment.from_wav(audio_url)
    segment=sound[start:end]
    response = create_segment(0.0,0.0,segment, 'speech')
    assert response['status_code']==200

def test_generate_target():
    audio_url='C:\\Users\\dell\\Downloads\\Music\\audio.wav'
    start=5*1000
    end=10*1000
    sound = AudioSegment.from_wav(audio_url)
    segment=sound[start:end]
    response = generate_target(segment)
    assert response['status_code']==200
 
def test_detect_language():
    source_langauge,target_lanaguge=detect_language('arabic','english')
    assert source_langauge is not None
    assert target_lanaguge is not None

def test_text_translation():
    target_text=text_translation('text','source','target')
    assert target_text is not None


def test_audio_speech_nonspeech_detection():
    speech_segments, non_speech_segments = audio_speech_nonspeech_detection("audio_url")
    assert len(speech_segments) > 0
    assert len(non_speech_segments) > 0

def test_speech_to_speech_translation():
    audio_url = "audio_url"
    source_language="english"
    target_language="arabic"
    response=speech_to_speech_translation(audio_url,source_language,target_language)
    assert response.status_code == 200
    assert response.json() == {"status_code":"succcessfully"}
 


if __name__ == "__main__":
    pytest.main([__file__])
