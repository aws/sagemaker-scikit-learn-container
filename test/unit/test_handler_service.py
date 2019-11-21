# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from mock import patch
import numpy as np
import pytest

from sagemaker_containers.beta.framework import (content_types, encoders, errors)
from sklearn.base import BaseEstimator

from sagemaker_sklearn_container.handler_service import HandlerService


handler = HandlerService().DefaultSKLearnUserModuleInferenceHandler()


@pytest.fixture(scope='module', name='np_array')
def fixture_np_array():
    return np.ones((2, 2))


class FakeEstimator(BaseEstimator):
    def __init__(self):
        pass

    @staticmethod
    def predict(input):
        return


@pytest.mark.parametrize(
    'json_data, expected', [
        ('[42, 6, 9]', np.array([42, 6, 9])),
        ('[42.0, 6.0, 9.0]', np.array([42., 6., 9.])),
        ('["42", "6", "9"]', np.array(['42', '6', '9'], dtype=np.float32)),
        (u'["42", "6", "9"]', np.array([u'42', u'6', u'9'], dtype=np.float32))])
def test_input_fn_json(json_data, expected):
    actual = handler.default_input_fn(json_data, content_types.JSON)
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    'csv_data, expected', [
        ('42\n6\n9\n', np.array([42, 6, 9], dtype=np.float32)),
        ('42.0\n6.0\n9.0\n', np.array([42., 6., 9.], dtype=np.float32)),
        ('42\n6\n9\n', np.array([42, 6, 9], dtype=np.float32))])
def test_input_fn_csv(csv_data, expected):
    deserialized_np_array = handler.default_input_fn(csv_data, content_types.CSV)
    assert np.array_equal(expected, deserialized_np_array)


@pytest.mark.parametrize('np_array', ([42, 6, 9], [42., 6., 9.]))
def test_input_fn_npz(np_array):
    input_data = encoders.array_to_npy(np_array)
    deserialized_np_array = handler.default_input_fn(input_data, content_types.NPY)

    assert np.array_equal(np_array, deserialized_np_array)

    float_32_array = np.array(np_array, dtype=np.float32)
    input_data = encoders.array_to_npy(float_32_array)
    deserialized_np_array = handler.default_input_fn(input_data, content_types.NPY)

    assert np.array_equal(float_32_array, deserialized_np_array)

    float_64_array = np.array(np_array, dtype=np.float64)
    input_data = encoders.array_to_npy(float_64_array)
    deserialized_np_array = handler.default_input_fn(input_data, content_types.NPY)

    assert np.array_equal(float_64_array, deserialized_np_array)


def test_input_fn_bad_content_type():
    with pytest.raises(errors.UnsupportedFormatError):
        handler.default_input_fn('', 'application/not_supported')


def test_default_model_fn():
    with pytest.raises(NotImplementedError):
        handler.default_model_fn('model_dir')


def test_predict_fn(np_array):
    mock_estimator = FakeEstimator()
    with patch.object(mock_estimator, 'predict') as mock:
        handler.default_predict_fn(np_array, mock_estimator)
    mock.assert_called_once()


def test_output_fn_json(np_array):
    response = handler.default_output_fn(np_array, content_types.JSON)
    assert response == (encoders.array_to_json(np_array.tolist()), content_types.JSON)


def test_output_fn_csv(np_array):
    response = handler.default_output_fn(np_array, content_types.CSV)
    assert response == ('1.0,1.0\n1.0,1.0\n', content_types.CSV)


def test_output_fn_npz(np_array):
    response = handler.default_output_fn(np_array, content_types.NPY)
    assert response == (encoders.array_to_npy(np_array), content_types.NPY)


def test_input_fn_bad_accept():
    with pytest.raises(errors.UnsupportedFormatError):
        handler.default_output_fn('', 'application/not_supported')
