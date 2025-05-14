# Stable3DGen

## Installation
Clone the repo:
```bash
git clone --recursive https://github.com/Stable-X/Stable3DGen.git
cd Stable3DGen
```

Create a conda environment (optional):
```bash
conda create -n stablex python=3.10
conda activate stablex
```

Install dependencies:
```bash
# pytorch (select correct CUDA version)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/{your-cuda-version}
pip install spconv-cu{your-cuda-version}==2.3.6 xformers==0.0.27.post2
# other dependencies
pip install -r requirements.txt
```

## Local Demo 🤗
Run by:
```bash
python app.py
```

<!-- License -->
## License
The model and code of Stable3DGen are adapted from [**Trellis**](https://github.com/microsoft/TRELLIS), which are licensed under the [MIT License](LICENSE). While the original Trellis is MIT licensed, we have specifically removed its dependencies on certain NVIDIA libraries (kaolin, nvdiffrast, flexicube) to ensure this adapted version can be used commercially. Stable3DGen itself is distributed under the [MIT License](LICENSE).

## Citation
If you find this work helpful, please consider citing our paper:
```
@article{ye2025hi3dgen,
  title={Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging},
  author={Ye, Chongjie and Wu, Yushuang and Lu, Ziteng and Chang, Jiahao and Guo, Xiaoyang and Zhou, Jiaqing and Zhao, Hao and Han, Xiaoguang},
  journal={arXiv preprint arXiv:2503.22236}, 
  year={2025}
}
```
