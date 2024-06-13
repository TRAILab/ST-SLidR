# Install
Tested with Python3.8 and CUDA 11.3.

## Clone VLM4AD 
```
git clone https://github.com/TRAILab/STSLidR.git
```

## Create and activate virtual environment 
```
conda create -n stslidr python=3.8
conda activate stslidr
```

## Install Pytorch + Torchvision compiled with CUDA 11.3:
```
pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Install Minkowski from source
```
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"` \
cd MinkowskiEngine
python setup.py install --force_cuda --blas=openblas
cd ..
```

## Install from pyproject.toml
`pip install .[torch]`
`pip install .`


## Datasets
Add symbolic link to nuScenes and SemanticKITTI datasets under datasets folder

```
ln -s /mnt/disks/wd_ssd/nuscences_dataset/data/sets/nuscenes/ ./datasets/nuscenes
ln -s /mnt/disks/wd_ssd/semantic_kitti/dataset/ ./datasets/kitti
```

## Superpixels
Generate SLIC superpixels in superpixels/nuscenes/superpixels_slic directory

```
python superpixel_segmenter.py