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

import importlib
import logging
import numpy as np
from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)


logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def default_model_fn(model_dir):
    """Loads a model. For Scikit-learn, a default function to load a model is not provided.
    Users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns: A Scikit-learn model.
    """
    return transformer.default_model_fn(model_dir)


def default_input_fn(input_data, content_type):
    """Takes request data and de-serializes the data into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:
            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
        The input_fn is responsible to take the request data and pre-process it before prediction.
    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.
    Returns:
        (obj): data ready for prediction.
    """
    np_array = encoders.decode(input_data, content_type)
    return np_array.astype(np.float32) if content_type in content_types.UTF8_TYPES else np_array


def default_predict_fn(input_data, model):
    """A default predict_fn for Scikit-learn. Calls a model on data deserialized in input_fn.
    Args:
        input_data: input data (Numpy array) for prediction deserialized by input_fn
        model: Scikit-learn model loaded in memory by model_fn
    Returns: a prediction
    """
    output = model.predict(input_data)
    return output


def default_output_fn(prediction, accept):
    """Function responsible to serialize the prediction for the response.
    Args:
        prediction (obj): prediction returned by predict_fn .
        accept (str): accept content-type expected by the client.
    Returns:
        (worker.Response): a Flask response object with the following args:
            * Args:
                response: the serialized data to return
                accept: the content-type that the data was transformed to.
    """
    return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)


def _user_module_transformer(user_module):
    model_fn = getattr(user_module, 'model_fn', default_model_fn)
    input_fn = getattr(user_module, 'input_fn', default_input_fn)
    predict_fn = getattr(user_module, 'predict_fn', default_predict_fn)
    output_fn = getattr(user_module, 'output_fn', default_output_fn)

    return transformer.Transformer(model_fn=model_fn, input_fn=input_fn, predict_fn=predict_fn,
                                   output_fn=output_fn)


def _user_module_execution_parameters_fn(user_module):
    return getattr(user_module, 'execution_parameters_fn', None)


def import_module(module_name, module_dir):

    try:  # if module_name already exists, use the existing one
        user_module = importlib.import_module(module_name)
    except ImportError:  # if the module has not been loaded, 'modules' downloads and installs it.
        user_module = modules.import_module(module_dir, module_name)
    except Exception:  # this shouldn't happen
        logger.info("Encountered an unexpected error.")
        raise

    user_module_transformer = _user_module_transformer(user_module)
    user_module_transformer.initialize()

    return user_module_transformer, _user_module_execution_parameters_fn(user_module)


app = None


def main(environ, start_response):
    global app

    if app is None:
        serving_env = env.ServingEnv()

        user_module_transformer, execution_parameters_fn = import_module(serving_env.module_name, 
                                                                         serving_env.module_dir)

        app = worker.Worker(transform_fn=user_module_transformer.transform,
                            module_name=serving_env.module_name,
                            execution_parameters_fn=execution_parameters_fn)

    return app(environ, start_response)
