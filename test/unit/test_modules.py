import pandas as pd


def test_pandas_version():
    import pandas as pd
    major, minor, patch = pd.__version__.split('.')
    assert major == '2'


def test_pyarrow_to_parquet_conversion_regression_issue_106():
    df = pd.DataFrame({'x': [1, 2]})
    df.to_parquet('test.parquet', engine='pyarrow')
