# SageMaker Scikit-learn Extension Container

The SageMaker Scikit-learn Extension Container is used in SageMaker Autopilot.

The SageMaker Scikit-learn Extension Container is built in 3 steps. The first 2 steps should be the same as building the [sagemaker-scikit-learn-container](https://github.com/aws/sagemaker-scikit-learn-container) image.

### Step 1: Base Image

The "base" Dockerfile encompass the installation of the framework and all of the dependencies needed.

Tagging scheme is based on <Scikit-learn_version>-cpu-py<python_version>. (e.g. 0.21.2-cpu-py3)

All "final" Dockerfiles build images using base images that use the tagging scheme above.

```
docker build -t sklearn-base:0.21.2-cpu-py3 -f docker/0.21.2/base/Dockerfile.cpu .
```

Notice that this Dockerfile has the updated version of sklearn (0.21.2) installed.

### Step 2: Final Image

The "final" Dockerfiles encompass the installation of the SageMaker specific support code.

All "final" Dockerfiles use base images for building.

These "base" images are specified with the naming convention of sklearn-base:<Scikit-learn_version>-cpu-py<python_version>.

Before building "final" images:

Build your "base" image. Make sure it is named and tagged in accordance with your "final" Dockerfile.

```
# Create the SageMaker Scikit-learn Container Python package.
python setup.py bdist_wheel
```

Then build the final image, like in the sagemaker-sklearn-container

```
docker build -t preprod-sklearn:0.21.2-cpu-py3 -f docker/0.21.2/final/Dockerfile.cpu .
```

### Step 3: Build the extension image for SageMaker Scikit-learn Extension Container

The "extension" Dockerfiles encompass the installation of the SageMaker Autopilot specific support code.

The "extension" Dockerfiles use final images for building.

Build the third additional Dockerfile needed for SageMaker Scikit-learn Extension Container. This Dockerfile specifies a hard dependecy on a certain version of scikit-learn (i.e. v0.21.2).

Tagging scheme is based on extension-0.1.0-cpu-py<python_version>. (e.g. extension-0.1.0-cpu-py3)

Before building the "extension" image:

Build your "base" and "final" images. Make sure the "extension" image is tagged in accordance with the prefix `extension` (i.e. `extension-0.1.0-cpu-py3`).

```
docker build -t preprod-sklearn:extension-0.1.0-cpu-py3 -f  docker/0.21.2/extension/Dockerfile.cpu .
```
