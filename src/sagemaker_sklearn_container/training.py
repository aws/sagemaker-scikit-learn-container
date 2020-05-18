# Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging

from sagemaker_training import entry_point, environment, runner

logger = logging.getLogger(__name__)


def train(training_environment):
    """Runs Scikit-learn training on a user supplied module in local SageMaker environment.
    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.

    Args:
        training_environment: training environment object containing environment variables,
                               training arguments and hyperparameters
    """
    logger.info('Invoking user training script.')
    entry_point.run(uri=training_environment.module_dir,
                    user_entry_point=training_environment.user_entry_point,
                    args=training_environment.to_cmd_args(),
                    env_vars=training_environment.to_env_vars(),
                    runner_type=runner.ProcessRunnerType)


def main():
    train(environment.Environment())
