<p align="center">
  <h2 align="center">No Pose, No Problem <img src="https://noposplat.github.io/static/images/icon.svg" width="20" style="position: relative; top: 1px;"> <br> Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images</h2>
 <p align="center">
    <a href="https://botaoye.github.io/">Botao Ye</a>
    ·
    <a href="https://sifeiliu.net/">Sifei Liu</a>
    ·
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    ·
    <a href="https://sunshineatnoon.github.io/">Xueting Li</a>
    ·
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a>
    ·
    <a href="https://faculty.ucmerced.edu/mhyang/">Ming-Hsuan Yang</a>
    ·
    <a href="https://pengsongyou.github.io/">Songyou Peng</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2410.24207">Paper</a> | <a href="https://noposplat.github.io/">Project Page</a> | <a href="#" style="color: grey; pointer-events: none; text-decoration: none;">Online Demo (Coming Soon)</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://noposplat.github.io/static/images/teaser.png" alt="Teaser" width="100%">
  </a>
</p>


<p align="center">
<strong>NoPoSplat</strong> predicts 3D Gaussians in a canonical space from unposed sparse images, <br> enabling high-quality novel view synthesis and accurate pose estimation.
</p>
<br>



<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#pre-trained-checkpoints">Pre-trained Checkpoints</a>
    </li>
    <li>
      <a href="#camera-conventions">Camera Conventions</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#running-the-code">Running the Code</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
</ol>
</details>

## Installation
Our code relies on Python 3.10+, and is developed based on PyTorch 2.1.2 and CUDA 11.8, but it should work with higher Pytorch/CUDA versions as well.

1. Clone NoPoSplat.
```bash
git clone https://github.com/cvg/NoPoSplat
cd NoPoSplat
```

2. Create the environment, here we show an example using conda.
```bash
conda create -y -n noposplat python=3.10
conda activate noposplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

3. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# NoPoSplat relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd src/model/encoder/backbone/croco/curope/
python setup.py build_ext --inplace
cd ../../../../../..
```

## Pre-trained Checkpoints
Our models are hosted on [Hugging Face](https://huggingface.co/botaoye/noposplat) 🤗

|                                                    Model name                                                    | Training resolutions | Training data |
|:----------------------------------------------------------------------------------------------------------------:|:--------------------:|:-------------:|
|                 [re10k.ckpt]( https://huggingface.co/botaoye/NoPoSplat/resolve/main/re10k.ckpt)                  |        256x256       |     re10k     |
|                  [acid.ckpt]( https://huggingface.co/botaoye/NoPoSplat/resolve/main/acid.ckpt )                  |        256x256       |     acid      |
|         [mixRe10kDl3dv.ckpt]( https://huggingface.co/botaoye/NoPoSplat/resolve/main/mixRe10kDl3dv.ckpt )         |        256x256       | re10k, dl3dv  |
| [mixRe10kDl3dv_512x512.ckpt]( https://huggingface.co/botaoye/NoPoSplat/resolve/main/mixRe10kDl3dv_512x512.ckpt ) |        512x512       | re10k, dl3dv  |

We assume the downloaded weights are located in the `pretrained_weights` directory.

## Camera Conventions
Our camera system is the same as [pixelSplat](https://github.com/dcharatan/pixelsplat). The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).
The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## Datasets
Please refer to [DATASETS.md](DATASETS.md) for dataset preparation.

## Running the Code
### Training
First download the [MASt3R](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth) pretrained model and put it in the `./pretrained_weights` directory.

Then call `src/main.py` via:

```bash
# 8 GPUs, with each batch size = 16. Remove the last two arguments if you don't want to use wandb for logging
python -m src.main +experiment=re10k wandb.mode=online wandb.name=re10k
```
This default training configuration requires 8x GPUs with a batch size of 16 on each GPU (>=80GB memory). 
The training will take approximately 6 hours to complete.
You can adjust the batch size to fit your hardware, but note that changing the total batch size may require modifying the initial learning rate to maintain performance.
You can refer to the [re10k_1x8](config/experiment/re10k_1x8.yaml) for training on 1 A6000 GPU (48GB memory), which will produce similar performance.


### Evaluation
#### Novel View Synthesis
```bash
# RealEstate10K
python -m src.main +experiment=re10k mode=test wandb.name=re10k dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json checkpointing.load=./pretrained_weights/re10k.ckpt test.save_image=true
# RealEstate10K
python -m src.main +experiment=acid mode=test wandb.name=acid dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json checkpointing.load=./pretrained_weights/acid.ckpt test.save_image=true
```
You can set `wandb.name=SAVE_FOLDER_NAME` to specify the saving path.

#### Pose Estimation
To evaluate the pose estimation performance, you can run the following command:
```bash
# RealEstate10K
python -m src.eval_pose +experiment=re10k +evaluation=eval_pose checkpointing.load=./pretrained_weights/mixRe10kDl3dv.ckpt dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json
# ACID
python -m src.eval_pose +experiment=acid +evaluation=eval_pose checkpointing.load=./pretrained_weights/mixRe10kDl3dv.ckpt dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json
# ScanNet-1500
python -m src.eval_pose +experiment=scannet_pose +evaluation=eval_pose checkpointing.load=./pretrained_weights/mixRe10kDl3dv.ckpt
```
Note that here we show the evaluation using the mixed model trained on RealEstate10K and DL3DV. You can replace the checkpoint path with other trained models.

## Acknowledgements
This project is developed with several fantastic repos: [pixelSplat](https://github.com/dcharatan/pixelsplat), [DUSt3R](https://github.com/naver/dust3r), and [CroCo](https://github.com/naver/croco). We thank the original authors for their excellent work.
We thank the kindly help of [David Charatan](https://davidcharatan.com/#/) for providing the evaluation code and the pretrained models for some of the previous methods.

## Citation

```
@article{ye2024noposplat,
      title   = {No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images},
      author  = {Ye, Botao and Liu, Sifei and Xu, Haofei and Xueting, Li and Pollefeys, Marc and Yang, Ming-Hsuan and Songyou, Peng},
      journal = {arXiv preprint arXiv:2410.24207},
      year    = {2024}
    }
```