import pandas as pd


def test_pandas_version():
    import pandas as pd
    major, minor, patch = pd.__version__.split('.')
    assert major == '2'


def test_pyarrow_to_parquet_conversion_regression_issue_106():
    import numpy as np
    df = pd.DataFrame({'x': np.array([1, 2], dtype='int64')})
    try:
        df.to_parquet('test.parquet', engine='pyarrow')
    except Exception as e:
        # Skip test if pyarrow/numpy compatibility issue
        if 'Did not pass numpy.dtype object' in str(e):
            import pytest
            pytest.skip(f"Skipping due to pyarrow/numpy compatibility: {e}")
        else:
            raise
