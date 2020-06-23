# Quant evaluator
## ImageNet
Only validation part of the ImageNet dataset is requried to run [comp_reram.py](./comp_reram.py) script.
Download ImageNet validation subset, extract to `~/datasets/imagenet/val`. Go to that folder and split
validation subset into class-specific sub-folders:
```bash
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x ./valprep.sh
./valprep.sh
```

## Python environment
Create python virtual environment and install PyTorch packages:
```bash
virtualenv ./env
source ./env/bin/activate
pip install torch torchvision matplotlib
```

The following versions are compatible with the script: `torch-1.5.1`, `torchvision-0.6.1` and `matplotlib-3.2.2`.
Confirmed with RTX 2080S and CUDA 10.2 (440.82).

## Running
```
cd ./comp_reram/
IMAGENET=~/datasets/imagenet python ./comp_reram.py
```

Optionally, directory for logs can be provided using `LOGDIR` environment variable.

## TODO
To package this script into docker container, we need to be able to specify cache directory for PyTorch models.
Be default, models are downloaded to `~/.cache/pytorch`.
