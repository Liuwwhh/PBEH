# PBEH
The source code for the paper "Prompt-driven Bit Extension Hashing".

## datasets & pre-trained cmh models
1. Download datasets MSCOCO and NUSWIDE

```
MSCOCO
url: https://pan.baidu.com/s/1uJ5DgDIJIBRownazZXOWnA?pwd=2025
code: 2025

NUSWIDE
url: https://pan.baidu.com/s/17Rn92JwYELzV4YNQ2bndmg?pwd=2025
code: 2025

MSCOCO-Imbalance
url: https://pan.baidu.com/s/1gzUoMh3P-hH2iNysMxWSBA?pwd=2025
code: 2025

NUSWIDE-Imbalance
url: https://pan.baidu.com/s/1njmBa0j0EfeD_CzT0V4ZgA?pwd=2025
code: 2025
```

2. Change the value of `data_path` in file `./config.yaml` to `/path/to/dataset`.

## python environment
``` bash
conda create -n PBEH python=3.8
conda activate PBEH
pip install -r requirements.txt
```

## training
``` python
python main.py
```
