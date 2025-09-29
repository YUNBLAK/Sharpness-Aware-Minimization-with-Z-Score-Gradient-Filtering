# Sharpness-Aware-Minimization-with-Z-Score-Gradient-Filtering
NeurIPS OPT 2025: Sharpness-Aware Minimization with Z-Score Gradient Filtering


### Dependencies

    Please check requirements.txt


### Tiny ImageNet Download
    
    mkdir -p data/tiny-imagenet
    cd data/tiny-imagenet
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip

### Auto Run

    bash run.sh

### Manual Run

    python main.py --dataset c10 --seed 52 --model-name res20s --sam --zsam
    python main.py --dataset c10 --seed 52 --model-name res20s --samsung
    python main.py --dataset c10 --seed 52 --model-name res20s --sam
    python main.py --dataset c10 --seed 52 --model-name res20s --fsam
    python main.py --dataset c10 --seed 52 --model-name res20s

### Args

    --dataset [c10, c100]


## [Error Handling] If you have "zero_gradient" error in Pytorch  
This error occurred because the zero_gradients function was removed due to a PyTorch version update. In recent versions of PyTorch (especially 2.0 and above), the following changes were made: The torch.autograd.gradcheck.zero_gradients() function is no longer supported. This function was available in earlier versions but has been removed in the latest releases. As a result, some parts of the advertorch library are not compatible with the latest PyTorch versions, causing this issue.


Summary, this happens because `zero_gradients` has been **removed** from PyTorch's `torch.autograd.gradcheck`.

### Solution
To fix this issue, manually define `zero_gradients` and replace its usage in the affected file.

#### 1. Remove the Import Statement
Open the file:

    /opt/conda/lib/python3.11/site-packages/advertorch/attacks/fast_adaptive_boundary.py

Find and **delete** the following line:

    from torch.autograd.gradcheck import zero_gradients

#### 2. Define zero_gradients Manually
At the top of the same file (fast_adaptive_boundary.py), add the following function:

    def zero_gradients(x):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
