# DGIR-Net

Evaluation code and pretrained weights for low-light image enhancement. The model is instantiated as `DGIRNet` from `net/DGIRNet.py`. Checkpoints saved before the DGDI rename are supported automatically.

## Setup

Install a matched CUDA-enabled PyTorch and torchvision pair for your GPU. From this repository run:

```bash
python -m pip install -r requirements-eval.txt
python -c "import torch; print(torch.cuda.is_available())"
```

The second command should print `True` for GPU testing. On the first paired evaluation, LPIPS needs torchvision's pretrained AlexNet weights in the torch hub cache or network access to download them.
## Datasets

The paired evaluation datasets are available separately from Baidu Netdisk: vhttps://pan.baidu.com/s/1WEFStozl27mmQTsjd6y-3w?pwd=uqhk 提取码: uqhk
## Evaluate

Point the two directory arguments to the actual folders containing the low-light images and their reference images. Each output directory must be new or empty. Benchmark data are supplied separately by the user.

```bash
python eval_gpu.py --dataset lolv1 --low-dir /data/LOL/eval15/low --gt-dir /data/LOL/eval15/high --output-dir results/lolv1
python eval_gpu.py --dataset lolv2-real --low-dir /data/LOLv2/Real_captured/Test/Low --gt-dir /data/LOLv2/Real_captured/Test/Normal --output-dir results/lolv2-real
python eval_gpu.py --dataset lolv2-syn --low-dir /data/LOLv2/Synthetic/Test/Low --gt-dir /data/LOLv2/Synthetic/Test/Normal --output-dir results/lolv2-syn
```


