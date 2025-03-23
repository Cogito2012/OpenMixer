# OpenMixer
This repository released the source code of OpenMixer (WACV 2025), heavily dependent on the [STMixer](https://github.com/MCG-NJU/STMixer) codebase.

## Installation
- Create conda environment:  
```bash
conda create -n openmixer python=3.7
```

- Install pytorch:  
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

- Install other libraries (including the OpenAI-CLIP):  
```bash
pip install -r requirements.txt
```

## Data Preparation
- First, please refer to the MMAction2 [JHMDB](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/jhmdb/README.md) and [UCF24](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ucf101_24/README.md) dataset preparation steps.

- Next, please download our released [Open-World splits](https://drive.google.com/drive/folders/1Bu5GNsGIfYD-4u_7WMjBOWZj_3zs-HbJ?usp=sharing). Make sure folders are structured as follows.
```bash
data
├──JHMDB
|   ├── openworld
|   ├── Frames
|   ├── JHMDB-MaskRCNN.pkl
|   ├── JHMDB-GT.pkl
├──UCF24
|   ├── openworld
|   ├── rgb-images
|   ├── UCF24-MaskRCNN.pkl
```

## Models

- Please download the pretrained `CLIP-ViP-B/16` checkpoint from [XPretrain/CLIP-ViP](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP), which is a video CLIP model served as the backbone of our model. After downloaded, make sure the file is located at `./pretrained/pretrain_clipvip_base_16.pt`.

- [Optional] We released three OpenMixer models and inference results for each of the JHMDB and UCF24 datasets here: [Google Drive](https://drive.google.com/drive/folders/1MDT_jcJolNZjuZ15cdhXyJmewMyVBKUP?usp=sharing). They correspond to the configurations in the folder `./config_files/`. Note that for the ZSR_ZSL setting, no model training needed. 


## Training

We provided an easy-to-use bash script to enable training and evaluation for different settings and datasets. For example, to train the OpenMixer model under the end-to-end setting on the JHMDB dataset using 4 specified GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash trainval.sh train jhmdb
```
Optionally, you may change the GPU IDs and dataset name to `ucf24`. For other settings, in the `trainval.sh`, change the `CFG_FILE` to `openmixer_zsr_tl.yaml` to train OpenMixer model under the ZSR+TL setting.


## Validation
We use the same bash script for validation (inference + evaluation)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash trainval.sh eval jhmdb
```
Optionally, you may change the GPU IDs and dataset name to `ucf24`. For other settings, in the `trainval.sh`, change the `CFG_FILE` to `openmixer_zsr_tl.yaml` and `openmixer_zsr_zsl.yaml` for evaluating models under the ZSR+TL and ZSR+ZSL settings, respectively.


## Acknowledgements
This project is built upon [STMixer](https://github.com/MCG-NJU/STMixer), [CLIP-ViP](https://github.com/microsoft/XPretrain/CLIP-ViP), and [OpenAI-CLIP](https://github.com/openai/CLIP). We sincerely thank contributors of all these great open-source repositories!


## Citation

If this project helps you in your research or project, please cite
our paper:

```
@InProceedings{bao2025wacv,
  title={Exploiting VLM Localizability and Semantics for Open Vocabulary Action Detection},
  author={Wentao Bao and Kai Li and Yuxiao Chen and Deep Patel and Martin Renqiang Min and Yu Kong},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```


