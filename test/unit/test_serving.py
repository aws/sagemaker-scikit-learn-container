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

from mock import patch, MagicMock
import numpy as np
import pytest
import os

from sklearn.base import BaseEstimator

from sagemaker_containers.beta.framework import (content_types, encoders, errors)
from sagemaker_sklearn_container import serving
from sagemaker_sklearn_container.exceptions import UserError
from sagemaker_sklearn_container.serving import default_model_fn, import_module


@pytest.fixture(scope='module', name='np_array')
def fixture_np_array():
    return np.ones((2, 2))


class FakeEstimator(BaseEstimator):
    def __init__(self):
        pass

    @staticmethod
    def predict(input):
        return


def dummy_execution_parameters_fn():
    return {'dummy': 'dummy'}


class DummyUserModule:
    def __init__(self):
        self.execution_parameters_fn = dummy_execution_parameters_fn

    def model_fn(self, model_dir):
        pass


@pytest.mark.parametrize(
    'json_data, expected', [
        ('[42, 6, 9]', np.array([42, 6, 9])),
        ('[42.0, 6.0, 9.0]', np.array([42., 6., 9.])),
        ('["42", "6", "9"]', np.array(['42', '6', '9'], dtype=np.float32)),
        (u'["42", "6", "9"]', np.array([u'42', u'6', u'9'], dtype=np.float32))])
def test_input_fn_json(json_data, expected):
    actual = serving.default_input_fn(json_data, content_types.JSON)
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    'csv_data, expected', [
        ('42\n6\n9\n', np.array([42, 6, 9], dtype=np.float32)),
        ('42.0\n6.0\n9.0\n', np.array([42., 6., 9.], dtype=np.float32)),
        ('42\n6\n9\n', np.array([42, 6, 9], dtype=np.float32))])
def test_input_fn_csv(csv_data, expected):
    deserialized_np_array = serving.default_input_fn(csv_data, content_types.CSV)
    assert np.array_equal(expected, deserialized_np_array)


@pytest.mark.parametrize('np_array', ([42, 6, 9], [42., 6., 9.]))
def test_input_fn_npz(np_array):
    input_data = encoders.array_to_npy(np_array)
    deserialized_np_array = serving.default_input_fn(input_data, content_types.NPY)

    assert np.array_equal(np_array, deserialized_np_array)

    float_32_array = np.array(np_array, dtype=np.float32)
    input_data = encoders.array_to_npy(float_32_array)
    deserialized_np_array = serving.default_input_fn(input_data, content_types.NPY)

    assert np.array_equal(float_32_array, deserialized_np_array)

    float_64_array = np.array(np_array, dtype=np.float64)
    input_data = encoders.array_to_npy(float_64_array)
    deserialized_np_array = serving.default_input_fn(input_data, content_types.NPY)

    assert np.array_equal(float_64_array, deserialized_np_array)


def test_input_fn_bad_content_type():
    with pytest.raises(errors.UnsupportedFormatError):
        serving.default_input_fn('', 'application/not_supported')


def test_default_model_fn():
    with pytest.raises(NotImplementedError):
        default_model_fn('model_dir')


def test_predict_fn(np_array):
    mock_estimator = FakeEstimator()
    with patch.object(mock_estimator, 'predict') as mock:
        serving.default_predict_fn(np_array, mock_estimator)
    mock.assert_called_once()


def test_output_fn_json(np_array):
    response = serving.default_output_fn(np_array, content_types.JSON)

    assert response.get_data(as_text=True) == encoders.array_to_json(np_array.tolist())
    assert response.content_type == content_types.JSON


def test_output_fn_csv(np_array):
    response = serving.default_output_fn(np_array, content_types.CSV)

    assert response.get_data(as_text=True) == '1.0,1.0\n1.0,1.0\n'
    assert content_types.CSV in response.content_type


def test_output_fn_npz(np_array):
    response = serving.default_output_fn(np_array, content_types.NPY)

    assert response.get_data() == encoders.array_to_npy(np_array)
    assert response.content_type == content_types.NPY


def test_input_fn_bad_accept():
    with pytest.raises(errors.UnsupportedFormatError):
        serving.default_output_fn('', 'application/not_supported')


@patch("sagemaker_sklearn_container.serving.transformer")
def test_user_module_transformer_with_transform_and_other_fn(mock_transformer):
    mock_module = MagicMock(spec=["model_fn", "transform_fn", "input_fn"])
    with pytest.raises(UserError):
        serving._user_module_transformer(mock_module)


@patch("sagemaker_sklearn_container.serving.transformer")
def test_user_module_transformer_with_transform_and_no_other_fn(mock_transformer):
    mock_module = MagicMock(spec=["model_fn", "transform_fn"])
    serving._user_module_transformer(mock_module)
    mock_transformer.Transformer.assert_called_once_with(
        model_fn=mock_module.model_fn, transform_fn=mock_module.transform_fn
    )


@patch('importlib.import_module')
def test_import_module_execution_parameters(importlib_module_mock):
    importlib_module_mock.return_value = DummyUserModule()
    _, execution_parameters_fn = import_module('dummy_module', 'dummy_dir')

    assert execution_parameters_fn == dummy_execution_parameters_fn


@patch('sagemaker_sklearn_container.serving.server')
def test_serving_entrypoint_start_gunicorn(mock_server):
    mock_server.start = MagicMock()
    serving.serving_entrypoint()
    mock_server.start.assert_called_once()


@patch.dict(os.environ, {'SAGEMAKER_MULTI_MODEL': 'True', })
@patch('sagemaker_sklearn_container.serving.start_model_server')
def test_serving_entrypoint_start_mms(mock_start_model_server):
    serving.serving_entrypoint()
    mock_start_model_server.assert_called_once()
