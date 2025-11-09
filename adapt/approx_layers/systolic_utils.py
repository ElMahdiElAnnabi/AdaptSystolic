# adapt/systolic_utils.py -

import torch
import torch.nn as nn
from typing import Tuple
from .layers_systolic import AdaPT_Conv2d_Systolic, AdaPT_Linear_Systolic
from .systolic_build import precompile_systolic_extensions

# --- helpers -----------------------------------------------------------------

def _is_linear_like(m: nn.Module) -> bool:
    if isinstance(m, nn.Linear):
        return True
    w = getattr(m, "weight", None)
    if w is not None and w.ndim == 2 and "linear" in m.__class__.__name__.lower():
        return True
    return False

def _get_linear_params(m: nn.Module) -> Tuple[int, int, bool]:
    in_f  = m.in_features if hasattr(m, "in_features") else m.weight.shape[1]
    out_f = m.out_features if hasattr(m, "out_features") else m.weight.shape[0]
    bias  = (getattr(m, "bias", None) is not None)
    return in_f, out_f, bias

def _is_conv2d_like(m: nn.Module) -> bool:
    if isinstance(m, nn.Conv2d):
        return True
    w = getattr(m, "weight", None)
    if w is not None and w.ndim == 4 and "conv2d" in m.__class__.__name__.lower():
        return True
    return False

def _get_conv2d_params(m: nn.Module):
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
                     sa_rows: int = 16, sa_cols: int = 16, 
                     precompile: bool = False) -> nn.Module:


    for name, m in list(model.named_children()):
        if _is_linear_like(m):
            in_f, out_f, has_bias = _get_linear_params(m)
            new_m = AdaPT_Linear_Systolic(
                size_in=in_f, size_out=out_f, bias=has_bias,
                axx_mult=axx_mult, use_exact=use_exact,
                sa_rows=sa_rows, sa_cols=sa_cols,   # << pass dims
            )
            new_m.weight.data.copy_(m.weight.data)
            if has_bias and getattr(m, "bias", None) is not None:
                new_m.bias.data.copy_(m.bias.data)
            setattr(model, name, new_m)
            continue

        if _is_conv2d_like(m):
            in_ch, out_ch, ksize, stride, pad, dil, groups, has_bias, p_mode = _get_conv2d_params(m)
            new_m = AdaPT_Conv2d_Systolic(
                in_channels=in_ch, out_channels=out_ch, kernel_size=ksize,
                stride=stride, padding=pad, dilation=dil, groups=groups,
                bias=has_bias, padding_mode=p_mode,
                axx_mult=axx_mult, use_exact=use_exact,
                sa_rows=sa_rows, sa_cols=sa_cols,   # << pass dims
            )
            new_m.weight.data.copy_(m.weight.data)
            if has_bias and getattr(m, "bias", None) is not None:
                new_m.bias.data.copy_(m.bias.data)
            setattr(model, name, new_m)
            continue

        # recurse (keep passing the dims)
        swap_to_systolic(m, use_exact=use_exact, axx_mult=axx_mult,
                         sa_rows=sa_rows, sa_cols=sa_cols, precompile=False)
    return model
# --- evaluation helpers ------------------------------------------------------

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
    NOW OPTIMIZED: Pre-compiles both variants before creating models.

    Args:
        model_fp      : trained baseline model (with loaded weights)
        model_factory : callable that returns a NEW, same-arch model
        axx_mult      : which multiplier variant to use (string)
        device        : torch device

    Returns:
        exact_model, approx_model
    """
    # PRE-COMPILE both variants ONCE
    print("Pre-compiling systolic extensions...")
    precompile_systolic_extensions(
        axx_mult=axx_mult,
        use_exact_variants=[True, False],  # Both exact and approx
        verbose=False
    )
    
    model_fp = model_fp.to(device)

    # Fresh copies of the baseline weights
    exact_model  = model_factory().to(device)
    exact_model.load_state_dict(model_fp.state_dict(), strict=True)

    approx_model = model_factory().to(device)
    approx_model.load_state_dict(model_fp.state_dict(), strict=True)

    # Wrap into systolic (now instant because pre-compiled)
    print("Converting to systolic (exact)...")
    exact_model  = swap_to_systolic(exact_model,  use_exact=True,  axx_mult=axx_mult, precompile=False)
    
    print("Converting to systolic (approx)...")
    approx_model = swap_to_systolic(approx_model, use_exact=False, axx_mult=axx_mult, precompile=False)

    return exact_model, approx_model

def compare_exact_vs_approx(model_fp, test_loader, model_factory, axx_mult="mul8s_acc", device="cpu"):
    """
    Compare baseline, exact systolic, and approx systolic models.
    NOW OPTIMIZED: Pre-compiles extensions once at the start.
    
    Args:
        model_fp      : trained model (already loaded with weights)
        test_loader   : DataLoader over the test set
        model_factory : callable that returns a NEW, same-arch model
        axx_mult      : multiplier variant
        device        : torch device
    """
    # PRE-COMPILE ONCE at the start
    print("=" * 70)
    print("Pre-compiling systolic extensions (one-time cost)...")
    print("=" * 70)
    import time
    t0 = time.time()
    precompile_systolic_extensions(
        axx_mult=axx_mult,
        use_exact_variants=[True, False],
        verbose=True
    )
    print(f"Pre-compilation took: {time.time()-t0:.2f}s\n")
    
    model_fp = model_fp.to(device)

    # 1) Baseline (normal execution)
    print("Evaluating baseline...")
    t0 = time.time()
    acc_baseline = eval_partial_dataset(model_fp, test_loader, device=device)
    print(f"  Baseline accuracy: {acc_baseline:.4f} ({time.time()-t0:.2f}s)\n")

    # 2) Clone weights into fresh models from factory
    print("Creating exact systolic model...")
    t0 = time.time()
    exact_model  = model_factory().to(device)
    exact_model.load_state_dict(model_fp.state_dict(), strict=True)
    exact_model  = swap_to_systolic(exact_model,  use_exact=True,  axx_mult=axx_mult, precompile=False)
    print(f"  Model creation: {time.time()-t0:.2f}s")
    
    print("Evaluating exact systolic...")
    t0 = time.time()
    exact_model.eval()
    acc_exact = eval_partial_dataset(exact_model,  test_loader, device=device)
    print(f"  Exact accuracy: {acc_exact:.4f} ({time.time()-t0:.2f}s)\n")

    # 3) Approx model
    print("Creating approx systolic model...")
    t0 = time.time()
    approx_model = model_factory().to(device)
    approx_model.load_state_dict(model_fp.state_dict(), strict=True)
    approx_model = swap_to_systolic(approx_model, use_exact=False, axx_mult=axx_mult, precompile=False)
    print(f"  Model creation: {time.time()-t0:.2f}s")
    
    print("Evaluating approx systolic...")
    t0 = time.time()
    approx_model.eval()
    acc_approx = eval_partial_dataset(approx_model, test_loader, device=device)
    print(f"  Approx accuracy: {acc_approx:.4f} ({time.time()-t0:.2f}s)\n")

    delta = acc_exact - acc_approx
    
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Baseline:      {acc_baseline:.4f}")
    print(f"Exact systolic: {acc_exact:.4f}")
    print(f"Approx systolic: {acc_approx:.4f}")
    print(f"Delta (exact - approx): {delta:.4f}")
    print("=" * 70)
    
    return acc_baseline, acc_exact, acc_approx, delta
