## 1. Setup

### Recommended Installation

For easy installation, use [conda](https://www.anaconda.com/):

```shell
conda create -n torch python=3.9
conda activate torch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip3 install numba nuscenes-devkit
```

Our method is based on [torchpack](https://github.com/zhijian-liu/torchpack) and [torchsparse](https://github.com/mit-han-lab/torchsparse). To install torchpack, we recommend to firstly install openmpi and mpi4py.

```shell
conda install -c conda-forge mpi4py openmpi
```

Install torchpack

```shell
pip install git+https://github.com/zhijian-liu/torchpack.git
```

Before installing torchsparse, install [Google Sparse Hash](https://github.com/sparsehash/sparsehash) library first.

```shell
sudo apt install libsparsehash-dev
```

Then install torchsparse (v1.4.0) by

```shell
pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

## 2. Data preparation

### nuScenes\_lidarseg

Download nuScenes full dataset (v1.0) and nuScene_lidarseg [here](https://www.nuscenes.org/download).

Assume your nuScenes is `~/work/dataset/nuscenes`. Edit the data root in `configs/nuscenes/default.yaml` as 

`~/work/dataset/nuscenes`.

## 3. Train

```shell
CUDA_VISIBLE_DEVICES=0,1,2 torchpack dist-run -np 3 python3 train_nusc_vcd.py configs/nuscenes/pretrain/vcd.yaml --run-dir runs/vcd
```
