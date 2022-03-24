# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import encodings
import json
import os
import subprocess
import sys
import time

import pytest
import requests

PING_URL = 'http://localhost:8080/ping'
INVOCATION_URL = 'http://localhost:8080/models/{}/invoke'
MODELS_URL = 'http://localhost:8080/models'
DELETE_MODEL_URL = 'http://localhost:8080/models/{}'

path = os.path.abspath(__file__)
resource_path = os.path.join(os.path.dirname(path), '..', 'resources')


@pytest.fixture(scope='session', autouse=True)
def volume():
    try:
        model_dir = os.path.join(resource_path, 'models')
        subprocess.check_call(
            'docker volume create --name dynamic_endpoint_model_volume --opt type=none '
            '--opt device={} --opt o=bind'.format(model_dir).split())
        yield model_dir
    finally:
        subprocess.check_call('docker volume rm dynamic_endpoint_model_volume'.split())


@pytest.fixture(scope='session', autouse=True)
def modulevolume():
    try:
        module_dir = os.path.join(resource_path, 'module')
        subprocess.check_call(
            'docker volume create --name dynamic_endpoint_module_volume --opt type=none '
            '--opt device={} --opt o=bind'.format(module_dir).split())
        yield module_dir
    finally:
        subprocess.check_call('docker volume rm dynamic_endpoint_module_volume'.split())


@pytest.fixture(scope='module', autouse=True)
def container(request, docker_base_name, tag):
    test_name = 'sagemaker-sklearn-serving-test'
    module_dir = os.path.join(resource_path, 'module')
    model_dir = os.path.join(resource_path, 'models')
    try:
        command = (
            'docker run --name {} -p 8080:8080'
            #' --mount type=volume,source=dynamic_endpoint_model_volume,target=/opt/ml/model,readonly'
            #' --mount type=volume,source=dynamic_endpoint_module_volume,target=/user_module,readonly'
            ' -v {}:/opt/ml/model'
            ' -v {}:/user_module'
            ' -e SAGEMAKER_BIND_TO_PORT=8080'
            ' -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999'
            ' -e SAGEMAKER_MULTI_MODEL=true'
            ' -e SAGEMAKER_PROGRAM={}'
            ' -e SAGEMAKER_SUBMIT_DIRECTORY={}'
            ' {}:{} serve'
        ).format(test_name, model_dir, module_dir, 'script.py', "/user_module/user_code.tar.gz", docker_base_name, tag)

        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        attempts = 0
        while attempts < 5:
            time.sleep(3)
            try:
                requests.get('http://localhost:8080/ping')
                break
            except Exception:
                attempts += 1
                pass

        yield proc.pid
    finally:
        subprocess.check_call('docker rm -f {}'.format(test_name).split())


def make_invocation_request(data, model_name, content_type='text/csv'):
    headers = {
        'Content-Type': content_type,
    }
    response = requests.post(INVOCATION_URL.format(model_name), data=data, headers=headers)
    return response.status_code, json.loads(response.content.decode(encodings.utf_8.getregentry().name))


def make_list_model_request():
    response = requests.get(MODELS_URL)
    return response.status_code, json.loads(response.content.decode(encodings.utf_8.getregentry().name))


def make_get_model_request(model_name):
    response = requests.get(MODELS_URL + '/{}'.format(model_name))
    return response.status_code, json.loads(response.content.decode(encodings.utf_8.getregentry().name))


def make_load_model_request(data, content_type='application/json'):
    headers = {
        'Content-Type': content_type
    }
    response = requests.post(MODELS_URL, data=data, headers=headers)
    return response.status_code, response.content.decode(encodings.utf_8.getregentry().name)


def make_unload_model_request(model_name):
    response = requests.delete(DELETE_MODEL_URL.format(model_name))
    return response.status_code, response.content.decode(encodings.utf_8.getregentry().name)


def test_ping():
    res = requests.get(PING_URL)
    assert res.status_code == 200


def test_list_models_empty():
    code, res = make_list_model_request()
    # assert code == 200
    assert res == {'models': []}


def test_delete_unloaded_model():
    # unloads the given model/version, no-op if not loaded
    model_name = 'non-existing-model'
    code, res = make_unload_model_request(model_name)
    assert code == 404


def test_load_and_unload_model():
    model_name = 'pickled-model-1'
    model_data = {
        'model_name': model_name,
        'url': '/opt/ml/model/{}'.format(model_name)
    }
    code, res = make_load_model_request(json.dumps(model_data))
    assert code == 200, res
    res_json = json.loads(res)
    assert res_json['status'] == 'Workers scaled'

    code, res = make_invocation_request('0.0, 0.0, 0.0, 0.0, 0.0, 0.0', model_name)
    assert code == 200, res

    code, res = make_unload_model_request(model_name)
    assert code == 200, res
    res_json = json.loads(res)
    assert res_json['status'] == "Model \"{}\" unregistered".format(model_name), res

    code, res = make_invocation_request('0.0, 0.0, 0.0, 0.0, 0.0, 0.0', model_name)
    assert code == 404, res
    assert res['message'] == "Model not found: {}".format(model_name), res


def test_load_and_unload_two_models():
    model_name_0 = 'pickled-model-1'
    model_data_0 = {
        'model_name': model_name_0,
        'url': '/opt/ml/model/{}'.format(model_name_0)
    }
    code, res = make_load_model_request(json.dumps(model_data_0))
    assert code == 200, res
    res_json = json.loads(res)
    assert res_json['status'] == 'Workers scaled'

    model_name_1 = 'pickled-model-2'
    model_data_1 = {
        'model_name': model_name_1,
        'url': '/opt/ml/model/{}'.format(model_name_1)
    }
    code, res = make_load_model_request(json.dumps(model_data_1))
    assert code == 200, res
    res_json = json.loads(res)
    assert res_json['status'] == 'Workers scaled'

    code, res = make_invocation_request('0.0, 0.0, 0.0, 0.0, 0.0, 0.0', model_name_0)
    assert code == 200, res

    code, res = make_invocation_request('0.0, 0.0, 0.0, 0.0, 0.0, 0.0', model_name_1)
    assert code == 200, res

    code, res = make_unload_model_request(model_name_0)
    assert code == 200, res
    res_json = json.loads(res)
    assert res_json['status'] == "Model \"{}\" unregistered".format(model_name_0), res

    code, res = make_unload_model_request(model_name_1)
    assert code == 200, res
    res_json = json.loads(res)
    assert res_json['status'] == "Model \"{}\" unregistered".format(model_name_1), res


def test_container_start_invocation_fail():
    x = {
        'instances': [1.0, 2.0, 5.0]
    }
    code, res = make_invocation_request(json.dumps(x), 'half_plus_three')
    assert code == 404
    assert res['message'] == "Model not found: {}".format('half_plus_three')


def test_load_one_model_two_times():
    model_name = 'pickled-model-1'
    model_data = {
        'model_name': model_name,
        'url': '/opt/ml/model/{}'.format(model_name)
    }
    code_load, res = make_load_model_request(json.dumps(model_data))
    assert code_load == 200, res
    res_json = json.loads(res)
    assert res_json['status'] == 'Workers scaled'

    code_load, res = make_load_model_request(json.dumps(model_data))
    assert code_load == 409
    res_json = json.loads(res)
    assert res_json['message'] == 'Model {} is already registered.'.format(model_name)
