# RL Quant
DNN mixed-precision quantization with reinforcement learning, derived from [HAQ](https://github.com/mit-han-lab/haq). 

## ReRAM Accelerator Mixed-Precision Quantization

### Running RL Search

Use `run_reram.sh` to run RL search. 
```{bash}
./run_reram.sh
```
This will call `reram_search.py` and start RL search. `reram_search.py` script calls `comp_reram/comp_reram.py` script to evaluate the performance a certain quantization scheme. 

### Evaluating Quantization Configuration
To evaluate the loss or accuracy of a specific quantization configuration, use `comp_reram/comp_reram.py` script. 
```{bash}
python3 comp_reram/comp_reram.py
```

### Testing Environment
- Ubuntu 18.04.4 LTS
- Python 3.6.9
- PyTorch 1.4.0
- GPU: NVIDIA Tesla P100
