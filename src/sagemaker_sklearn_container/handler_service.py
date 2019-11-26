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
from __future__ import absolute_import
import numpy as np
import textwrap

from sagemaker_inference import content_types, decoder, default_inference_handler, encoder
from sagemaker_inference.default_handler_service import DefaultHandlerService

from sagemaker_sklearn_container.mms_patch.mms_transformer import Transformer


class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on the type MXNet model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md
    """

    class DefaultSKLearnUserModuleInferenceHandler(default_inference_handler.DefaultInferenceHandler):

        @staticmethod
        def default_model_fn(model_dir):
            """Loads a model. For Scikit-learn, a default function to load a model is not provided.
            Users should provide customized model_fn() in script.
            Args:
                model_dir: a directory where model is saved.
            Returns: A Scikit-learn model.
            """
            raise NotImplementedError(textwrap.dedent("""
            Please provide a model_fn implementation.
            See documentation for model_fn at https://github.com/aws/sagemaker-python-sdk
            """))

        @staticmethod
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
            np_array = decoder.decode(input_data, content_type)
            if len(np_array.shape) == 1:
                np_array = np_array.reshape(1, -1)
            return np_array.astype(np.float32) if content_type in content_types.UTF8_TYPES else np_array

        @staticmethod
        def default_predict_fn(input_data, model):
            """A default predict_fn for Scikit-learn. Calls a model on data deserialized in input_fn.
            Args:
                input_data: input data (Numpy array) for prediction deserialized by input_fn
                model: Scikit-learn model loaded in memory by model_fn
            Returns: a prediction
            """
            output = model.predict(input_data)
            return output

        @staticmethod
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
            return encoder.encode(prediction, accept), accept

    def __init__(self):
        transformer = Transformer(default_inference_handler=self.DefaultSKLearnUserModuleInferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)
