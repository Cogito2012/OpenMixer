## HilaCAM for CLIP Visual Attention

Please go to [gScoreCAM](https://github.com/anguyen8/gScoreCAM), download the folders `hila_clip/` and `pytorch_grad_cam/`, and put them in this folder.

The following commands show the steps:
```shell
cd alphaction/cam
git clone https://github.com/anguyen8/gScoreCAM
cp -r gScoreCAM/hila_clip gScoreCAM/pytorch_grad_cam .
rm -rf gScoreCAM

```

After that, please ensure the following python packages are installed:
```shell
pip install ttach kornia scikit-learn scikit-image
```