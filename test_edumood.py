import pytest
import pandas as pd
from edumood_recognizer import EduMoodSessionStats, EduMoodRecognizer

def test_session_stats_initialization():
    """Test that session stats initializes correctly"""
    stats = EduMoodSessionStats()
    assert stats.records == []
    assert len(stats.records) == 0

def test_session_stats_add_record():
    """Test adding records to session stats"""
    stats = EduMoodSessionStats()
    test_record = {"happy": 2, "sad": 1, "neutral": 3}
    
    stats.add_record(test_record)
    assert len(stats.records) == 1
    assert stats.records[0] == test_record

def test_session_stats_to_dataframe():
    """Test converting records to DataFrame"""
    stats = EduMoodSessionStats()
    stats.add_record({"happy": 2, "sad": 1})
    df = stats.to_dataframe()
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "happy" in df.columns
    assert df["happy"].iloc[0] == 2

def test_recognizer_initialization():
    """Test that recognizer initializes with correct parameters"""
    stats = EduMoodSessionStats()
    recognizer = EduMoodRecognizer(stats, analyze_every_n=5)
    
    assert recognizer.session_stats == stats
    assert recognizer.analyze_every_n == 5
    assert recognizer.frame_count == 0
    assert recognizer.latency_records == []

def test_empty_dataframe():
    """Test DataFrame conversion with no records"""
    stats = EduMoodSessionStats()
    df = stats.to_dataframe()
    
    assert df.empty
    assert list(df.columns) == [
        "recorded_at", "happy", "sad", "angry", "surprise", 
        "neutral", "disgusted", "fearful"
    ]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
