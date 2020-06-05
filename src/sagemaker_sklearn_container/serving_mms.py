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
import logging
from math import ceil
import multiprocessing
import os
from retrying import retry
from subprocess import CalledProcessError

from sagemaker_containers.beta.framework import env, modules

from sagemaker_sklearn_container import handler_service
from sagemaker_sklearn_container.mms_patch import model_server

HANDLER_SERVICE = handler_service.__name__

PORT = 8080
DEFAULT_MAX_CONTENT_LEN = 6 * 1024 ** 2
MAX_CONTENT_LEN_LIMIT = 20 * 1024 ** 2
MMS_NUM_MODEL_WORKERS_INIT = 1
MMS_MODEL_JOB_QUEUE_SIZE_DEFAULT = 100


def get_mms_config_file_path():
    return os.environ['SKLEARN_MMS_CONFIG']


def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)


@retry(stop_max_delay=1000 * 30, retry_on_exception=_retry_if_error)
def _start_model_server(is_multi_model, handler):
    # there's a race condition that causes the model server command to
    # sometimes fail with 'bad address'. more investigation needed
    # retry starting mms until it's ready
    logging.info("Trying to set up model server handler: {}".format(handler))
    _set_mms_configs(is_multi_model, handler)
    model_server.start_model_server(handler_service=handler,
                                    is_multi_model=is_multi_model,
                                    config_file=get_mms_config_file_path())


def _set_default_if_not_exist(sagemaker_env_var_name, default_value):
    if not os.getenv(sagemaker_env_var_name, None):
        os.environ[sagemaker_env_var_name] = str(default_value)


def _set_mms_configs(is_multi_model, handler):
    """Set environment variables for MMS to parse during server initialization. These env vars are used to
    propagate the config.properties.tmp file used during MxNet Model Server initialization.
    'SAGEMAKER_MMS_MODEL_STORE' has to be set to the model location during single model inference because MMS
    is initialized with the model. In multi-model mode, MMS is started with no models loaded.
    Note: Ideally, instead of relying on env vars, this should be written directly to a config file.
    """
    max_content_length = os.getenv("MAX_CONTENT_LENGTH", DEFAULT_MAX_CONTENT_LEN)
    if int(max_content_length) > MAX_CONTENT_LEN_LIMIT:
        # Cap at 20mb
        max_content_length = MAX_CONTENT_LEN_LIMIT

    max_workers = multiprocessing.cpu_count()
    max_job_queue_size = 2 * max_workers

    # Max heap size = (max workers + max job queue size) * max payload size * 1.2 (20% buffer) + 128 (base amount)
    max_heap_size = ceil((max_workers + max_job_queue_size) * (int(max_content_length) / 1024 ** 2) * 1.2) + 128

    os.environ["SAGEMAKER_MMS_MODEL_STORE"] = '/'
    os.environ["SAGEMAKER_MMS_LOAD_MODELS"] = ''
    os.environ["SAGEMAKER_MMS_DEFAULT_HANDLER"] = handler

    # Users can define port
    _set_default_if_not_exist("SAGEMAKER_BIND_TO_PORT", str(PORT))

    # Multi Model Server configs, exposed to users as env vars
    _set_default_if_not_exist("SAGEMAKER_NUM_MODEL_WORKERS", MMS_NUM_MODEL_WORKERS_INIT)
    _set_default_if_not_exist("SAGEMAKER_MODEL_JOB_QUEUE_SIZE", MMS_MODEL_JOB_QUEUE_SIZE_DEFAULT)
    _set_default_if_not_exist("SAGEMAKER_MAX_REQUEST_SIZE", max_content_length)

    # JVM configurations for MMS, exposed to users as env vars
    _set_default_if_not_exist("SAGEMAKER_MAX_HEAP_SIZE", str(max_heap_size) + 'm')
    _set_default_if_not_exist("SAGEMAKER_MAX_DIRECT_MEMORY_SIZE", os.environ["SAGEMAKER_MAX_HEAP_SIZE"])

    MMS_CONFIG_FILE_PATH = get_mms_config_file_path()

    # TODO: Revert config.properties.tmp to config.properties and add back in vmargs
    # set with environment variables after MMS implements parsing environment variables
    # for vmargs, update MMS section of final/Dockerfile.cpu to match, and remove the
    # following code.
    try:
        with open(MMS_CONFIG_FILE_PATH + '.tmp', 'r') as f:
            with open(MMS_CONFIG_FILE_PATH, 'w+') as g:
                g.write("vmargs=-XX:-UseLargePages"
                        + " -XX:+UseParNewGC"
                        + " -XX:MaxMetaspaceSize=32M"
                        + " -XX:InitiatingHeapOccupancyPercent=25"
                        + " -Xms" + os.environ["SAGEMAKER_MAX_HEAP_SIZE"]
                        + " -Xmx" + os.environ["SAGEMAKER_MAX_HEAP_SIZE"]
                        + " -XX:MaxDirectMemorySize=" + os.environ["SAGEMAKER_MAX_DIRECT_MEMORY_SIZE"] + "\n")
                g.write(f.read())
    except Exception:
        pass


def start_model_server():
    serving_env = env.ServingEnv()
    is_multi_model = True

    modules.import_module(serving_env.module_dir, serving_env.module_name)
    _start_model_server(is_multi_model, HANDLER_SERVICE)
