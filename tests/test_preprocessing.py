# tests/test_preprocessing.py
import pandas as pd
from scripts.preprocess import preprocess_data


def test_preprocess_data_output_type():
    dummy_data = pd.DataFrame({
        'sepal_length': [5.1],
        'sepal_width': [3.5],
        'petal_length': [1.4],
        'petal_width': [0.2]
    })
    processed_data = preprocess_data(dummy_data)
    assert isinstance(processed_data, pd.DataFrame)


def test_preprocess_data_columns():
    dummy_data = pd.DataFrame({
        'sepal_length': [5.1],
        'sepal_width': [3.5],
        'petal_length': [1.4],
        'petal_width': [0.2]
    })
    processed_data = preprocess_data(dummy_data)
    expected_columns = ['sepal_length', 'sepal_width',
                        'petal_length', 'petal_width']
    assert list(processed_data.columns) == expected_columns


def test_preprocess_data_no_side_effects():
    original_data = pd.DataFrame({
        'sepal_length': [5.1],
        'sepal_width': [3.5],
        'petal_length': [1.4],
        'petal_width': [0.2]
    })

    data_copy = original_data.copy(deep=True)
    preprocess_data(data_copy)
    pd.testing.assert_frame_equal(original_data, original_data)
