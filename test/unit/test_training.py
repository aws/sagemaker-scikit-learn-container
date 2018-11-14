# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from mock import MagicMock, patch

from sagemaker_sklearn_container import training


def mock_training_env(current_host='algo-1', module_dir='s3://my/script', module_name='svm', **kwargs):
    return MagicMock(current_host=current_host, module_dir=module_dir, module_name=module_name, **kwargs)


@patch('sagemaker_containers.beta.framework.modules.run_module')
def test_single_machine(run_module):
    env = mock_training_env()
    training.train(env)

    run_module.assert_called_with('s3://my/script', env.to_cmd_args(), env.to_env_vars(), 'svm')
