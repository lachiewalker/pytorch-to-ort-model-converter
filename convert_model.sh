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

if [[ $# != 2 ]]; then
    echo "Incorrect number of arguments supplied. Please check usage instructions."
    exit 1
fi

FILE=$1
if [[ ! -f "$FILE" ]]; then
    echo "Model file not found, please check your input for typos".
    exit 1
fi

if [[ "$(docker images -q convert-mnv2)" == "" ]]; then
	echo "Docker image: convert-mnv2 not found. Building image from Dockerfile..."

	docker build -t convert-mnv2 .
else
	echo "Docker image:convert-mnv2 found."
fi

# recopy the py after the buiild, so you aren't running on old code. 

echo "Spinning up container..."

docker run --name model_converter -dit convert-mnv2

echo "Copying PyTorch model to container..."

docker cp $1 model_converter:/opt/$1


echo "Copying completed."
echo "Converting model to .onnx..."

docker exec model_converter python3 ./opt/convert_to_onnx.py "/opt/$1" $2

#docker cp model_converter:/opt/${strarr[0]}.log "${strarr[0]}.log"

echo "Converting model to .ort..."

docker exec model_converter python3 -m onnxruntime.tools.convert_onnx_models_to_ort "/opt/${strarr[0]}.onnx" 

echo "Copying .ort file to host machine..."

docker cp model_converter:/opt/${strarr[0]}.ort "${strarr[0]}.ort"

echo "Killing container..."

docker stop model_converter > /dev/null
docker rm model_converter > /dev/null

echo "done."
