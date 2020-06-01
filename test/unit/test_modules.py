def test_pandas_version():
    import pandas as pd
    major, minor, patch = pd.__version__.split('.')
    assert major == '1'
