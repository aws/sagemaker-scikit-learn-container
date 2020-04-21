# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker_training import runner
from sagemaker_sklearn_container import training


def mock_training_env(current_host='algo-1', module_dir='s3://my/script', user_entry_point='svm', **kwargs):
    return MagicMock(current_host=current_host, module_dir=module_dir, user_entry_point=user_entry_point, **kwargs)


@patch('sagemaker_training.entry_point.run')
def test_single_machine(run_entry_point):
    env = mock_training_env()
    training.train(env)

    run_entry_point.assert_called_with(uri=env.module_dir,
                                       user_entry_point=env.user_entry_point,
                                       args=env.to_cmd_args(),
                                       env_vars=env.to_env_vars(),
                                       runner_type=runner.ProcessRunnerType)
