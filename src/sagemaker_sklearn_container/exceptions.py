# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import warnings


# TODO: Consolidate this and other modules into a common package consumed by both XGB and SKLearn
class BaseToolkitError(Exception):
    """Abstract base for all errors that may cause an algorithm to exit/terminate
    unsuccessfully. All direct sub-classes should be kept/maintained in this file.
    These errors are grouped into three categories:
        1. AlgorithmError: an unexpected or unknown failure that cannot be
                           avoided by the user and is due to a bug in the
                           algorithm.
        2. UserError:      a failure which can be prevented/avoided by the
                           user (e.g. change mini_batch_size).
        3. PlatformError:  a failure due to an environmental requirement not
                           being met (e.g. if the /opt/ml/training directory
                           is missing).
    Args: see `Attributes` below.
    Attributes:
        message     (string): Description of why this exception was raised.
        caused_by   (exception): The underlying exception that caused this
                                 exception to be raised. This should be a
                                 non-BaseToolkitError.
    """

    def __init__(self, message=None, caused_by=None):
        formatted_message = BaseToolkitError._format_exception_message(message, caused_by)
        super(BaseToolkitError, self).__init__(formatted_message)
        self.message = formatted_message
        self.caused_by = caused_by

    @staticmethod
    def _format_exception_message(message, caused_by):
        """Generates the exception message.
        If a message has been explicitly passed then we use that as the exception
        message. If we also know the underlying exception type we prepend that
        to the name.
        If there is no message but we have an underlying exception then we use
        that exceptions message and prepend the type of the exception.
        """
        if message:
            formatted_message = message
        elif caused_by:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress deprecation warning
                formatted_message = getattr(caused_by, "message", str(caused_by))
        else:
            formatted_message = "unknown error occurred"

        if caused_by:
            formatted_message += f" (caused by {caused_by.__class__.__name__})"

        return formatted_message


class AlgorithmError(BaseToolkitError):
    """Exception used to indicate a problem that occurred with the algorithm."""

    def __init__(self, message=None, caused_by=None):
        super(AlgorithmError, self).__init__(message, caused_by)


class UserError(BaseToolkitError):
    """Exception used to indicate a problem caused by mis-configuration or other user input."""

    def __init__(self, message=None, caused_by=None):
        super(UserError, self).__init__(message, caused_by)


class PlatformError(BaseToolkitError):
    """Exception used to indicate a problem caused by the underlying platform (e.g. network time-outs)."""

    def __init__(self, message=None, caused_by=None):
        super(PlatformError, self).__init__(message, caused_by)
