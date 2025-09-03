# adapt/systolic_utils.py
import torch
import torch.nn as nn
from copy import deepcopy
# adapt/systolic_utils.py

import torch.nn as nn
from typing import Tuple
from .layers_systolic import AdaPT_Conv2d_Systolic, AdaPT_Linear_Systolic

# --- helpers -----------------------------------------------------------------

def _is_linear_like(m: nn.Module) -> bool:
    # True for nn.Linear, AdaPT_Linear, quantized linears, etc.
    if isinstance(m, nn.Linear):
        return True
    # Heuristic: has 2D weight and attr name contains 'Linear'
    w = getattr(m, "weight", None)
    if w is not None and w.ndim == 2 and "linear" in m.__class__.__name__.lower():
        return True
    return False

def _get_linear_params(m: nn.Module) -> Tuple[int, int, bool]:
    # Works for nn.Linear and most Linear-likes
    in_f  = m.in_features if hasattr(m, "in_features") else m.weight.shape[1]
    out_f = m.out_features if hasattr(m, "out_features") else m.weight.shape[0]
    bias  = (getattr(m, "bias", None) is not None)
    return in_f, out_f, bias

def _is_conv2d_like(m: nn.Module) -> bool:
    # True for nn.Conv2d, AdaPT_Conv2d, quant_nn.Conv2d, etc.
    if isinstance(m, nn.Conv2d):
        return True
    # Heuristic: weight is 4D and class name contains 'Conv2d'
    w = getattr(m, "weight", None)
    if w is not None and w.ndim == 4 and "conv2d" in m.__class__.__name__.lower():
        return True
    return False

def _get_conv2d_params(m: nn.Module):
    # Pull common Conv2d hyper-params safely
    in_ch  = getattr(m, "in_channels",  m.weight.shape[1] * getattr(m, "groups", 1))
    out_ch = getattr(m, "out_channels", m.weight.shape[0])
    ksize  = getattr(m, "kernel_size",  (m.weight.shape[2], m.weight.shape[3]))
    stride = getattr(m, "stride",       (1, 1))
    pad    = getattr(m, "padding",      (0, 0))
    dil    = getattr(m, "dilation",     (1, 1))
    groups = getattr(m, "groups",       1)
    bias   = (getattr(m, "bias", None) is not None)
    p_mode = getattr(m, "padding_mode", "zeros")
    return in_ch, out_ch, ksize, stride, pad, dil, groups, bias, p_mode

# --- main swap ---------------------------------------------------------------

def swap_to_systolic(model: nn.Module, use_exact: bool, axx_mult: str = 'mul8s_acc',
                     sa_rows: int = 16, sa_cols: int = 16) -> nn.Module:
    """
    Recursively replace Linear/Conv2d *and* AdaPT_Linear/AdaPT_Conv2d (and similar)
    with AdaPT_Linear_Systolic / AdaPT_Conv2d_Systolic.
    """
    # If wrapped with DP/DDP, descend into .module
    if hasattr(model, "module") and isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        inner = swap_to_systolic(model.module, use_exact, axx_mult, sa_rows, sa_cols)
        model.module = inner
        return model

    for name, m in list(model.named_children()):
        # --- Linear-like ------------------------------------------------------
        if _is_linear_like(m):
            in_f, out_f, has_bias = _get_linear_params(m)
            new_m = AdaPT_Linear_Systolic(
                size_in=in_f, size_out=out_f, bias=has_bias,
                axx_mult=axx_mult, use_exact=use_exact
            )
            # copy weights/bias
            new_m.weight.data.copy_(m.weight.data)
            if has_bias and getattr(m, "bias", None) is not None:
                new_m.bias.data.copy_(m.bias.data)
            setattr(model, name, new_m)
            continue

        # --- Conv2d-like ------------------------------------------------------
        if _is_conv2d_like(m):
            in_ch, out_ch, ksize, stride, pad, dil, groups, has_bias, p_mode = _get_conv2d_params(m)
            new_m = AdaPT_Conv2d_Systolic(
                in_channels=in_ch, out_channels=out_ch, kernel_size=ksize,
                stride=stride, padding=pad, dilation=dil, groups=groups,
                bias=has_bias, padding_mode=p_mode,
                axx_mult=axx_mult, use_exact=use_exact
            )
            new_m.weight.data.copy_(m.weight.data)
            if has_bias and getattr(m, "bias", None) is not None:
                new_m.bias.data.copy_(m.bias.data)
            setattr(model, name, new_m)
            continue

        # --- Recurse into submodules -----------------------------------------
        swap_to_systolic(m, use_exact=use_exact, axx_mult=axx_mult, sa_rows=sa_rows, sa_cols=sa_cols)

    return model


@torch.no_grad()
def eval_top1(model, dataloader, device='cuda'):
    model.eval().to(device)
    correct = total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


import torch
from torch.autograd import Variable

# def check_equal(batch_size=3, seed=0):
#     layer = AdaPT_Linear_Systolic(size_in=7, size_out=5, bias=True, axx_mult='mul8s_acc', use_exact=True)
    
#     torch.manual_seed(seed)
#     in_features = layer.weight.shape[1]
#     out_features = layer.weight.shape[0]

#     # Random input batch
#     x = torch.randint(-128, 128, (batch_size, in_features), dtype=torch.float32)

#     # Quantize inputs/weights
#     with torch.no_grad():
#         amax_x = x.abs().max()
#         amax_w = layer.weight.abs().max()
#         layer.quantizer.amax = amax_x
#         layer.quantizer_w.amax = amax_w

#         qx = layer.quantizer(x).to(torch.int8)                # [B,K]
#         qw = layer.quantizer_w(layer.weight).to(torch.int8)   # [N,K]

#         # Run systolic C++ kernel
#         out_cpp = layer.axx_linear_kernel.forward(qx, qw)     # [B,N] int32

#         # Reference: PyTorch int32 matmul
#         out_ref = (qx.to(torch.int32) @ qw.to(torch.int32).t())  # [B,N]

#         # Compare
#         equal = torch.equal(out_cpp, out_ref)
#         print(f"Equal: {equal}")
#         if not equal:
#             diff = (out_cpp - out_ref).abs()
#             print(f"  Max diff: {diff.max().item()}")
#         return equal

# def testlinear():

#     layer = AdaPT_Linear_Systolic(size_in=7, size_out=5, bias=True, axx_mult='mul8s_acc', use_exact=True)

#     x = torch.randn(4, 7)  # [B,K]
#     y = layer(x)

#     assert y.shape == (4, 5)
#     assert y.dtype == torch.float32
#     print("Smoke OK")
    
def eval_whole_dataset(model, dataloader, device="cpu"):
    model.eval().to(device)
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return correct / max(1, total)

def eval_partial_dataset(model, dataloader, device="cpu", fraction=0.05):
    """
    Evaluate model on only a fraction of the dataset (default 5%).
    """
    model.eval().to(device)
    correct = total = 0
    num_batches = len(dataloader)
    max_batches = max(1, int(num_batches * fraction))

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(dataloader):
            if b_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()

    return correct / max(1, total)

def make_exact_and_approx_models(model_fp, model_factory, axx_mult="mul8s_acc", device="cpu"):
    """
    Returns two models (exact systolic and approx systolic) that share weights from model_fp.

    Args:
        model_fp      : trained baseline model (with loaded weights)
        model_factory : callable that returns a NEW, same-arch model
                        e.g., lambda: resnet50(pretrained=False, axx_mult=axx_mult)
        axx_mult      : which multiplier variant to use (string)
        device        : torch device

    Returns:
        exact_model, approx_model
    """
    model_fp = model_fp.to(device)

    # Fresh copies of the baseline weights
    exact_model  = model_factory().to(device)
    exact_model.load_state_dict(model_fp.state_dict(), strict=True)

    approx_model = model_factory().to(device)
    approx_model.load_state_dict(model_fp.state_dict(), strict=True)

    # Wrap into systolic
    exact_model  = swap_to_systolic(exact_model,  use_exact=True,  axx_mult=axx_mult)
    approx_model = swap_to_systolic(approx_model, use_exact=False, axx_mult=axx_mult)

    return exact_model, approx_model


def compare_exact_vs_approx(model_fp, test_loader, model_factory, axx_mult="mul8s_acc", device="cpu"):
    """
    model_fp      : trained model (already loaded with weights)
    test_loader   : DataLoader over the test set
    model_factory : callable that returns a NEW, same-arch model (uninitialized weights)
                    e.g., lambda: resnet50(pretrained=False, axx_mult=axx_mult)
    """
    model_fp = model_fp.to(device)

    # 1) Baseline (normal execution)
    print("Evaluating baseline")
    acc_baseline = eval_partial_dataset(model_fp, test_loader, device=device)

    # 2) Clone weights into fresh models from factory
    exact_model  = model_factory().to(device)
    exact_model.load_state_dict(model_fp.state_dict(), strict=True)

    approx_model = model_factory().to(device)
    approx_model.load_state_dict(model_fp.state_dict(), strict=True)

    # 3) Wrap into systolic (exact) and evaluate
    exact_model  = swap_to_systolic(exact_model,  use_exact=True,  axx_mult=axx_mult)
    print("Evaluating exact systolic")
    exact_model.eval()
    acc_exact    = eval_partial_dataset(exact_model,  test_loader, device=device)

    # 4) Wrap into systolic (approx) and evaluate
    approx_model = swap_to_systolic(approx_model, use_exact=False, axx_mult=axx_mult)
    print("Evaluating approx systolic")
    approx_model.eval()
    acc_approx   = eval_partial_dataset(approx_model, test_loader, device=device)

    delta = acc_exact - acc_approx
    return acc_baseline, acc_exact, acc_approx, delta