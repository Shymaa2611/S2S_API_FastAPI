import pytest
from fastapi.testclient import TestClient
from main import *
from unittest.mock import Mock

client = TestClient(app)

def test_create_segment():
    response = create_segment(0.0,0.0, 'segment', 'speech')
    assert response.status_code == 200
    
def test_generate_target():
    response = client.post("/generate_target/", json={"audio": "audio_data"})
    assert response.status_code == 200

def test_audio_speech_nonspeech_detection():
    speech_segments, non_speech_segments = audio_speech_nonspeech_detection("audio_url")
    assert len(speech_segments) > 0
    assert len(non_speech_segments) > 0

def test_speech_to_speech_translation():
    audio_url = "audio_url"
    source_language="english"
    target_language="arabic"
    response = client.post("/speech_translation/", json={"audio_url": audio_url,"source_language":source_language,"target_language":target_language})
    assert response.status_code == 200
    assert response.json() == {"status_code":"succcessfully"}





if __name__ == "__main__":
    pytest.main([__file__])
