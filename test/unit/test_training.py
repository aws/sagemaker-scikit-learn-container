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

import sagemaker_containers.beta.framework as framework
from sagemaker_sklearn_container import training


def mock_training_env(current_host='algo-1', module_dir='s3://my/script', user_entry_point='svm', **kwargs):
    return MagicMock(current_host=current_host, module_dir=module_dir, user_entry_point=user_entry_point, **kwargs)


@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_single_machine(run_entry_point, download_and_install):
    env = mock_training_env()
    training.train(env)

    download_and_install.assert_called_with(env.module_dir)

    run_entry_point.assert_called_with(env.module_dir, env.user_entry_point, env.to_cmd_args(), env.to_env_vars(),
                                       runner=framework.runner.ProcessRunnerType)
