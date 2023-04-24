#!/bin/bash

# --------------------------------------------------------------
#
# 	This script starts and manages a docker container to
# 	convert PyTorch .pth models to ONNXRuntime's .ort
#	format.
#
#	USAGE: . convert_model [pytorch_model_file] [N]
#
#	...where N = number of species identified by the model
#
#	Eg: . convert_model model257species.pyt 257
#
#	2pi Software
#	Lachlan Walker
#
# --------------------------------------------------------------

readarray -d . -t strarr <<< "$1"

echo "Loading docker image..."

docker build -t convert-mnv2 .

echo "Spinning up container..."

docker run --name model_converter -dit convert-mnv2

echo "Copying PyTorch model to container..."

docker cp $1 model_converter:/opt/$1

echo "Copying completed."
echo "Converting model to .onnx..."

docker exec model_converter ./opt/convert_to_onnx $1 $2

echo "Success!"
echo "Converting model to .ort..."

docker exec model_converter python3 -m onnxruntime.tools.convert_onnx_models_to_ort "/opt/${strarr[0]}.onnx" 

echo "Success!"
echo "Copying .ort file to host machine..."

docker cp model_converter:/opt/${strarr[0]}.ort "${strarr[0]}.ort"

echo "Success!"
echo "Killing container..."

docker stop model_converter > /dev/null
docker rm model_converter > /dev/null

echo "done."
