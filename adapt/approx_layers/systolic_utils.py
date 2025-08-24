# adapt/systolic_utils.py
import torch
import torch.nn as nn
from copy import deepcopy
from .layers_systolic import AdaPT_Conv2d_Systolic, AdaPT_Linear_Systolic

def swap_to_systolic(model: nn.Module, use_exact: bool, axx_mult: str = 'mul8s_acc'):
    """Recursively replace Conv/Linear-like AdaPT layers with systolic versions.
       Set use_exact=True for exact multipliers, False for LUT approximate."""
    for name, m in list(model.named_children()):
        # Linear
        if isinstance(m, nn.Linear):
            new_m = AdaPT_Linear_Systolic(
                m.in_features, m.out_features, bias=(m.bias is not None),
                axx_mult=axx_mult, use_exact=use_exact
            )
            # copy weights/bias
            new_m.weight.data.copy_(m.weight.data)
            if new_m.bias is not None and m.bias is not None:
                new_m.bias.data.copy_(m.bias.data)
            setattr(model, name, new_m)

        # Conv2d
        elif isinstance(m, nn.Conv2d):
            new_m = AdaPT_Conv2d_Systolic(
                m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding,
                m.dilation, m.groups, bias=(m.bias is not None), padding_mode='zeros',
                axx_mult=axx_mult, use_exact=use_exact
            )
            new_m.weight.data.copy_(m.weight.data)
            if new_m.bias is not None and m.bias is not None:
                new_m.bias.data.copy_(m.bias.data)
            setattr(model, name, new_m)

        else:
            swap_to_systolic(m, use_exact=use_exact, axx_mult=axx_mult)
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
import torch

import torch

import torch

def check_equal(batch_size=3, seed=0):
    layer = AdaPT_Linear_Systolic(size_in=7, size_out=5, bias=True, axx_mult='mul8s_acc', use_exact=True)
    
    torch.manual_seed(seed)
    in_features = layer.weight.shape[1]
    out_features = layer.weight.shape[0]

    # Random input batch
    x = torch.randint(-128, 128, (batch_size, in_features), dtype=torch.float32)

    # Quantize inputs/weights
    with torch.no_grad():
        amax_x = x.abs().max()
        amax_w = layer.weight.abs().max()
        layer.quantizer.amax = amax_x
        layer.quantizer_w.amax = amax_w

        qx = layer.quantizer(x).to(torch.int8)                # [B,K]
        qw = layer.quantizer_w(layer.weight).to(torch.int8)   # [N,K]

        # Run systolic C++ kernel
        out_cpp = layer.axx_linear_kernel.forward(qx, qw)     # [B,N] int32

        # Reference: PyTorch int32 matmul
        out_ref = (qx.to(torch.int32) @ qw.to(torch.int32).t())  # [B,N]

        # Compare
        equal = torch.equal(out_cpp, out_ref)
        print(f"Equal: {equal}")
        if not equal:
            diff = (out_cpp - out_ref).abs()
            print(f"  Max diff: {diff.max().item()}")
        return equal

def testlinear():

    layer = AdaPT_Linear_Systolic(size_in=7, size_out=5, bias=True, axx_mult='mul8s_acc', use_exact=True)

    x = torch.randn(4, 7)  # [B,K]
    y = layer(x)

    assert y.shape == (4, 5)
    assert y.dtype == torch.float32
    print("Smoke OK")
    
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

def compare_exact_vs_approx(model_fp, test_loader, model_factory, axx_mult="mul8s_acc", device="cpu"):
    """
    model_fp      : trained model (already loaded with weights)
    test_loader   : DataLoader over the test set
    model_factory : callable that returns a NEW, same-arch model (uninitialized weights)
                    e.g., lambda: resnet50(pretrained=False, axx_mult=axx_mult)
    """
    model_fp = model_fp.to(device)

    # 1) Baseline (normal execution)
    acc_baseline = eval_whole_dataset(model_fp, test_loader, device=device)

    # 2) Clone weights into fresh models from factory
    exact_model  = model_factory().to(device)
    exact_model.load_state_dict(model_fp.state_dict(), strict=True)

    approx_model = model_factory().to(device)
    approx_model.load_state_dict(model_fp.state_dict(), strict=True)

    # 3) Wrap into systolic (exact) and evaluate
    exact_model  = swap_to_systolic(exact_model,  use_exact=True,  axx_mult=axx_mult)
    acc_exact    = eval_whole_dataset(exact_model,  test_loader, device=device)

    # 4) Wrap into systolic (approx) and evaluate
    approx_model = swap_to_systolic(approx_model, use_exact=False, axx_mult=axx_mult)
    acc_approx   = eval_whole_dataset(approx_model, test_loader, device=device)

    delta = acc_exact - acc_approx
    return acc_baseline, acc_exact, acc_approx, delta

# def compare_exact_vs_approx(model_fp, dataloader, axx_mult="mul8s_acc", device=None):
#     """
#     Compare accuracy of:
#       1. Baseline model (plain CPU inference)
#       2. Exact systolic model
#       3. Approx systolic model

#     Assumes quantizers (amax) are already calibrated.
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Ensure model is in eval mode
#     model_fp.eval()

#     # --- 1. Baseline (plain CPU model) ---
#     baseline_model = type(model_fp)(*getattr(model_fp, "init_args", []))
#     baseline_model.load_state_dict(model_fp.state_dict())
#     baseline_model.eval()
#     acc_baseline = eval_top1(baseline_model, dataloader, device="cpu")  # force CPU

#     # --- 2. Exact systolic ---
#     exact_model = type(model_fp)(*getattr(model_fp, "init_args", [])).to(device)
#     exact_model.load_state_dict(model_fp.state_dict())
#     exact_model.eval()
#     exact_model = swap_to_systolic(exact_model, use_exact=True, axx_mult=axx_mult)
#     acc_exact = eval_top1(exact_model, dataloader, device=device)

#     # --- 3. Approx systolic ---
#     approx_model = type(model_fp)(*getattr(model_fp, "init_args", [])).to(device)
#     approx_model.load_state_dict(model_fp.state_dict())
#     approx_model.eval()
#     approx_model = swap_to_systolic(approx_model, use_exact=False, axx_mult=axx_mult)
#     acc_approx = eval_top1(approx_model, dataloader, device=device)

#     # --- Deltas ---
#     delta_exact_vs_baseline = acc_baseline - acc_exact
#     delta_approx_vs_baseline = acc_baseline - acc_approx
#     delta_exact_vs_approx   = acc_exact - acc_approx

#     return {
#         "baseline": acc_baseline,
#         "exact": acc_exact,
#         "approx": acc_approx,
#         "Δ_exact_vs_baseline": delta_exact_vs_baseline,
#         "Δ_approx_vs_baseline": delta_approx_vs_baseline,
#         "Δ_exact_vs_approx": delta_exact_vs_approx,
#     }

# def compare_exact_vs_approx(model_fp, dataloader, axx_mult="mul8s_acc", device=None):
#     """
#     Compare accuracy of exact systolic vs approximate systolic model.
#     Assumes quantizers (amax) are already calibrated in the modules.
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Move original model to device & eval
#     model_fp = model_fp.to(device)
#     model_fp.eval()

#     # --- Clone models using state_dict ---
#     model_class = type(model_fp)

#     # Exact model
#     exact_model = model_class(*getattr(model_fp, "init_args", [])).to(device)
#     exact_model.load_state_dict(model_fp.state_dict())

#     # Approx model
#     approx_model = model_class(*getattr(model_fp, "init_args", [])).to(device)
#     approx_model.load_state_dict(model_fp.state_dict())

#     # --- Wrap into systolic (exact) ---
#     exact_model = swap_to_systolic(exact_model, use_exact=True, axx_mult=axx_mult)
#     acc_exact = eval_top1(exact_model, dataloader, device=device)

#     # --- Wrap into systolic (approx) ---
#     approx_model = swap_to_systolic(approx_model, use_exact=False, axx_mult=axx_mult)
#     acc_approx = eval_top1(approx_model, dataloader, device=device)

#     delta = acc_exact - acc_approx
#     return acc_exact, acc_approx, delta
