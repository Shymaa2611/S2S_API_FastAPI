import pytest
from fastapi.testclient import TestClient
from main import app, create_segment, generate_target, audio_speech_nonspeech_detection, split_audio_segments, text_to_text_translation, speech_to_text_process, text_to_speech, construct_audio

client = TestClient(app)

def test_create_segment():
    response = client.post("/create_segment/", json={"start_time": 0.0, "end_time": 1.0, "audio": "audio_data", "type": "speech"})
    assert response.status_code == 200
    

def test_generate_target():
    response = client.post("/generate_target/", json={"audio": "audio_data"})
    assert response.status_code == 200

def test_audio_speech_nonspeech_detection():
    speech_segments, non_speech_segments = audio_speech_nonspeech_detection("audio_url")
    assert len(speech_segments) > 0
    assert len(non_speech_segments) > 0



def test_text_to_text_translation():
    translated_text = text_to_text_translation("english_text")
    assert translated_text is not None

def test_speech_to_text_process():
    text = speech_to_text_process("audio_data")
    assert text is not None


if __name__ == "__main__":
    pytest.main([__file__])
