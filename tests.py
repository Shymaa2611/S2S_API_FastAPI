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

def test_text_to_text_translation():
    translated_text = en_text_to_ar_text_translation("english_text")
    assert translated_text is not None

def test_speech_to_text_process():
    text = en_speech_to_en_text_process("audio_data")
    assert text is not None

def test_speech_to_speech_translation_en_ar():
    audio_url = "audio_url"
    response = client.post("/speech2speech/", json={"audio_url": audio_url})
    assert response.status_code == 200
    assert response.json() == {"status_code":"succcessfully"}

def test_construct_audio(mock_generate_target, mock_AudioSegment, mock_Session):
    session_instance = mock_Session.return_value
    session_instance.query.return_value.order_by.return_value.all.return_value = [
        Mock(audio=b'audio1', start_time=1),
        Mock(audio=b'audio2', start_time=2),
        Mock(audio=b'audio3', start_time=3)
    ]
    mock_AudioSegment.return_value.empty.return_value = Mock()
    mock_AudioSegment.return_value.sum.return_value = Mock()
    construct_audio()
    mock_generate_target.assert_called_once_with(audio=mock_AudioSegment.return_value.sum.return_value)
    session_instance.query.return_value.delete.assert_called_once()
    session_instance.commit.assert_called_once()
    session_instance.close.assert_called_once()

def test_get_all_audio_segments(mock_Audio_segment, mock_Session):
    session_instance = mock_Session.return_value
    session_instance.query.return_value.all.return_value = [
        Mock(id=1, start_time=0, end_time=10, type='type', audio=b'audio_data')
    ]

    result = get_all_audio_segments()
    session_instance.query.assert_called_once()
    assert result == {
        "segments": [{
            "id": 1,
            "start_time": 0,
            "end_time": 10,
            "type": 'type',
            "audio_url": "segments//segment1_audio.wav"
        }]
    }

def test_extract_15_seconds(mock_AudioSegment):
    mock_audio_segment_instance = Mock()
    mock_AudioSegment.from_file.return_value = mock_audio_segment_instance
    mock_extracted_segment = Mock()
    mock_audio_segment_instance.__getitem__.return_value = mock_extracted_segment
    result = extract_15_seconds(b'audio_data', 0, 30)
    mock_AudioSegment.from_file.assert_called_once_with(Mock(), format='wav')
    mock_audio_segment_instance.__getitem__.assert_called_once_with(slice(0, 15000))
    mock_extracted_segment.export.assert_called_once_with('temp.wav', format='wav')
    assert result == 'temp.wav'

def test_get_audio(mock_translation, mock_Session):
    mock_translation.return_value = None
    session_instance = mock_Session.return_value
    target_audio_instance = Mock(audio=b'audio_data')
    session_instance.query.return_value.order_by.return_value.first.return_value = target_audio_instance
    result = get_ar_audio("audio_url")
    mock_translation.assert_called_once_with("audio_url")
    session_instance.query.assert_called_once()
    session_instance.query.return_value.delete.assert_called_once()
    session_instance.commit.assert_called_once()
    session_instance.close.assert_called_once()
    assert result == {"audio_url": "target_audio.wav"}


if __name__ == "__main__":
    pytest.main([__file__])
