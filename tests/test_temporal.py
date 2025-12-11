import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.analysis.temporal import TemporalSegmenter

@pytest.fixture
def sample_data():
    """Creates a synthetic dataframe spanning 6 months (2020-01 to 2020-06)"""
    dates = []
    # Dense month: Jan (60 items)
    dates.extend([datetime(2020, 1, 15)] * 60)
    # Sparse month: Feb (10 items)
    dates.extend([datetime(2020, 2, 15)] * 10)
    # Dense month: Mar (60 items)
    dates.extend([datetime(2020, 3, 15)] * 60)
    # Dense month: Apr (60 items)
    dates.extend([datetime(2020, 4, 15)] * 60)
    
    df = pd.DataFrame({"date": dates, "value": range(len(dates))})
    return df

def test_initialization(sample_data):
    seg = TemporalSegmenter(sample_data)
    assert seg.min_date == sample_data["date"].min()
    assert seg.max_date == sample_data["date"].max()

def test_rolling_window_logic(sample_data):
    """Test 3-month window, 1-month step"""
    seg = TemporalSegmenter(sample_data)
    windows = list(seg.generate_windows(window_months=3, step_months=1, min_count=0))
    
    # Expected Windows starting from 2020-01-01:
    # 1. Jan-Feb-Mar (End date Apr 1) -> Count: 60+10+60 = 130
    # 2. Feb-Mar-Apr (End date May 1) -> Count: 10+60+60 = 130
    # 3. Mar-Apr...
    
    w1 = windows[0]
    assert w1["start_date"] == datetime(2020, 1, 1)
    assert w1["end_date"] == datetime(2020, 4, 1)
    assert w1["count"] == 130
    
    w2 = windows[1]
    assert w2["start_date"] == datetime(2020, 2, 1)
    assert w2["end_date"] == datetime(2020, 5, 1)
    # Feb(10) + Mar(60) + Apr(60) = 130
    assert w2["count"] == 130

def test_density_filter(sample_data):
    """Test min_count filtering"""
    # Create very sparse data
    sparse_dates = [datetime(2020, 1, 1)] * 5 # Only 5 items total
    df = pd.DataFrame({"date": sparse_dates})
    
    seg = TemporalSegmenter(df)
    
    # Threshold 10, should return nothing
    windows = list(seg.generate_windows(window_months=3, min_count=10))
    assert len(windows) == 0
    
    # Threshold 1, should return
    windows = list(seg.generate_windows(window_months=3, min_count=1))
    assert len(windows) > 0

def test_string_date_conversion():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"]})
    seg = TemporalSegmenter(df)
    assert pd.api.types.is_datetime64_any_dtype(seg.df["date"])

if __name__ == "__main__":
    # verification run
    import sys
    sys.exit(pytest.main(["-v", __file__]))
