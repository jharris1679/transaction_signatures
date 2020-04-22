# Transaction Signatures
Predicting future spending behaviour of KOHO users.

### Steps to run training locally

Clone the repo and navigate to root directory.

`git clone https://github.com/jharris1679/transaction_signatures.git`

`cd transaction_signatures`

Running in Docker is encouraged. This repo uses the `nvidia/cuda:latest` image.
Provide an experiment ID and the name of the model directory to the build script.

`./build_local.sh [EXPERIMENT_ID] [MODEL_DIR]`

Currently there is only one model directory, `tx_model`. It orients inputs as one sequence per transaciton. 
In the future there will be another model that uses one sequence per user.

Once the container has finished buidling, it will provide a bash command prompt. 
To begin training, run

`./exec_training.sh`

It is configured to load the data samples included in this repo.

If Docker is not available, cuda and NVIDIA drivers will need to be installed manually to use GPU.
Python dependancies can be installed by running

`pip3 install --requirement tx_model/payload/requirements.txt`


