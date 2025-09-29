# NeurIPS 2025 OPT: Sharpness-Aware Minimization with Z-Score Gradient Filtering (Official Repository)
Author: Vincent-Daniel Yun (Juyoung Yun) <br>
Institute: University of Southern California<br>
[[Paper]](https://arxiv.org/abs/2505.02369) [[Workshop]](https://opt-ml.org/) [[Author Google Scholar]](https://scholar.google.com/citations?user=mlfYKfgAAAAJ&hl=en) <br>


<img width="1046" height="172" alt="Image" src="https://github.com/user-attachments/assets/330b8b9f-5f85-4943-90b9-daf77c98e038" />
<br> <br>



The settings of this repository are based on the default configuration of:
    
    https://github.com/omihub777/ViT-CIFAR


### Dependencies

    # Please check requirements.txt
    torchsummary
    pytorch-lightning==1.2.1
    comet-ml==3.3.5
    matplotlib
    numpy
    pandas
    scipy
    numpy==1.26.4
    scikit-learn
    warmup_scheduler

### Tiny ImageNet Download
    
    mkdir -p data/tiny-imagenet
    cd data/tiny-imagenet
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip

### Auto Run
    bash run.sh

### Manual Run

    # Ours (Zsharp)
    python main.py --dataset c10 --seed 52 --model-name res20s --sam --zsam

    # Adaptive SAM
    python main.py --dataset c10 --seed 52 --model-name res20s --samsung

    # Standard SAM
    python main.py --dataset c10 --seed 52 --model-name res20s --sam

    # Friendly SAM
    python main.py --dataset c10 --seed 52 --model-name res20s --fsam

    # AdamW
    python main.py --dataset c10 --seed 52 --model-name res20s

### Args

    --dataset [c10, c100]
    --model-name [res20s, res56s, vgg16_bn, vit]


# Adaptation to other codes

### Zsharp Class:

    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, hparams=None, **kwargs):
            assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super().__init__(params, defaults)
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)
            self.zsam_enabled = getattr(hparams, "zsam", False) if hparams else False
            self.zsam_alpha = getattr(hparams, "alpha", False) if hparams else False

        @torch.no_grad()
        def first_step(self, closure=None, zero_grad=False):
            grad_norm = self._grad_norm()

            for group in self.param_groups:
                rho = group["rho"]
                scale = rho / (grad_norm + 1e-12)
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    self.state[p]["old_p"] = p.data.clone()
                    grad = p.grad
                    if self.zsam_enabled and grad.ndim > 1:
                        grad_mean = grad.mean()
                        grad_std = grad.std() + 1e-12
                        z_norm_grad = (grad - grad_mean) / grad_std
                        threshold = torch.quantile(z_norm_grad.abs(), self.zsam_alpha) 
                        mask = z_norm_grad.abs() > threshold  
                        filtered_grad = grad.clone()
                        filtered_grad[~mask] = 0.0
                        e_w = rho * filtered_grad / (filtered_grad.norm(p=2) + 1e-12) if filtered_grad.norm(p=2) > 0 else grad * scale
                    else:
                        e_w = grad * scale
                    p.add_(e_w)

                    self.state[p]["prev_ew"] = e_w.clone()

            if zero_grad:
                self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.data = self.state[p]["old_p"]
            self.base_optimizer.step()
            if zero_grad:
                self.zero_grad()

        def step(self, closure=None):
            assert closure is not None, "SAM requires closure"
            closure = torch.enable_grad()(closure)
            self.first_step(closure=closure)
            closure()
            self.second_step()

        def _grad_norm(self):
            shared_device = self.param_groups[0]["params"][0].device
            norm = torch.norm(torch.stack([
                (torch.abs(p) if group["adaptive"] else 1.0) * p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2)
            return norm

    class Zharp(SAM):
        def __init__(self, params, hparams, rho=0.05, adaptive=False, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8):
            if hparams.sgd:
                base_opt = torch.optim.SGD
            else:
                base_opt = torch.optim.Adam
            super().__init__(
                params=params,
                base_optimizer=base_opt,
                rho=rho,
                adaptive=adaptive,
                hparams=hparams, 
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )

### Optimizer Block:

    optimizer = self.Zharp(
        self.model.parameters(),
        hparams=self.hparams,
        lr=self.hparams.lr,
        betas=(self.hparams.beta1, self.hparams.beta2),
        weight_decay=self.hparams.weight_decay
            )

<br><br>


## [Error Handling] If you have "zero_gradient" error in Pytorch  
This error occurs because the zero_gradients function was removed in recent PyTorch updates. In PyTorch 2.0 and above, the function

    torch.autograd.gradcheck.zero_gradients()

is no longer supported. It was available in earlier versions but has since been removed. As a result, some parts of the advertorch library are not compatible with the latest versions of PyTorch, leading to this issue.

In summary: the error happens because `zero_gradients` has been removed from `torch.autograd.gradcheck` in PyTorch.

<br>

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


<br><br><br>

# Acknowledgement
This research was supported by Brian Impact Foundation, a non-profit organization dedicated to the advancement of science and technology for all. 

# Reference

    @article{yun2025zsharp,
      title     = {Sharpness-Aware Minimization with Z-Score Gradient Filtering},
      author    = {Yun, Vincent-Daniel},
      journal   = {arXiv preprint arXiv:2505.02369},
      year      = {2025},
      doi       = {10.48550/arXiv.2505.02369},
      url       = {https://arxiv.org/abs/2505.02369},
    }



