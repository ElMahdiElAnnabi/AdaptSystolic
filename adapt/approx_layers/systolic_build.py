# adapt/approx_layers/systolic_build.py
from __future__ import annotations
import os, sys, glob, shutil, re, uuid
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable, List

from torch.utils.cpp_extension import load as cpp_load
import torch

__all__ = [
    "SystolicBuilder",
    "precompile_systolic_extensions",
    "clear_systolic_disk_cache",
    "list_systolic_binaries",
]

# -------------------- cache maintenance helpers --------------------

def _cache_root() -> str:
    return os.environ.get("TORCH_EXTENSIONS_DIR",
                          os.path.expanduser("~/.cache/torch_extensions/systolic"))

def clear_systolic_disk_cache(pattern: Optional[str] = None, *, verbose: bool = True):
    """
    Remove compiled systolic JIT artifacts from disk AND purge in-process modules.

    pattern: glob pattern for module names (e.g. '*mul8s_1L2H*'), or None for all.
    """
    root = _cache_root()
    if verbose:
        print(f"[clear_systolic_disk_cache] root={root} pattern={pattern or 'ALL'}")

    g = os.path.join(root, pattern or "systolic_*")
    for path in sorted(glob.glob(g)):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                if verbose: print("  rm -r", path)
            else:
                os.remove(path)
                if verbose: print("  rm", path)
        except Exception as e:
            if verbose: print("  ! failed to remove", path, "—", e)

    # purge Python’s view
    pat = re.compile((pattern or "systolic_").replace("*", ".*"))
    for m in list(sys.modules.keys()):
        if pat.match(m):
            if verbose: print(f"  del sys.modules['{m}']")
            del sys.modules[m]

    # purge torch cpp_extension registry
    try:
        from torch.utils import cpp_extension as _ce
        if hasattr(_ce, "_loaded_modules"):
            _ce._loaded_modules.clear()  # type: ignore[attr-defined]
            if verbose: print("  cleared torch cpp_extension _loaded_modules")
    except Exception:
        pass

def list_systolic_binaries() -> List[str]:
    """Return full paths of all compiled .so binaries in the cache root."""
    root = _cache_root()
    return sorted(glob.glob(os.path.join(root, "systolic_*.so")))

# -------------------------- builder --------------------------

class SystolicBuilder:
    """
    Compiles and caches CPU systolic kernels (C++ extensions) for Linear/Conv2d.

    - One compiled module per (op, axx_mult, use_exact, sa_rows, sa_cols)
    - Builds are persisted in a user-writable directory (default:
      ~/.cache/torch_extensions/systolic or $TORCH_EXTENSIONS_DIR)
    - Also writes an *unsuffixed alias* .so so plain imports by name succeed.
    """

    _compiled_modules: Dict[Tuple[str, str, bool, int, int], object]

    def __init__(
        self,
        *,
        verbose: bool = False,
        build_dir: Optional[str] = None,
        include_paths: Optional[Iterable[str]] = None,
        cflags: Optional[Iterable[str]] = None,
        ldflags: Optional[Iterable[str]] = None,
        sa_rows: int = 16,
        sa_cols: int = 16,
    ) -> None:
        self.verbose = verbose
        self._compiled_modules = {}

        self._this_dir = Path(__file__).resolve().parent
        self._cpu_kernels_dir = (self._this_dir.parent / "cpu-kernels").resolve()

        env_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
        default_dir = env_dir or os.path.expanduser("~/.cache/torch_extensions/systolic")
        self._build_dir = os.path.abspath(build_dir) if build_dir else default_dir

        default_cflags = [
            "-O3",
            "-march=native",
            "-fopenmp",
            "-fPIC",
            "-std=c++17",
        ]
        self._cflags = list(default_cflags) + list(cflags or [])
        self._ldflags = ["-fopenmp"] + list(ldflags or [])
        self._include_paths = list(include_paths or [])

        self._default_sources = {
            "linear": self._cpu_kernels_dir / "axx_linear_systolic.cpp",
            "conv2d": self._cpu_kernels_dir / "axx_conv2d_systolic.cpp",
        }

    # ----------------------- public API -----------------------

    def get(
        self,
        *,
        op: str,
        use_exact: bool,
        axx_mult: str,
        sa_rows: int = 16,
        sa_cols: int = 16,
        src: Optional[str] = None,
    ):
        key = (op, axx_mult, use_exact, sa_rows, sa_cols)
        if key in self._compiled_modules:
            return self._compiled_modules[key]
        return self.build(
            op=op,
            use_exact=use_exact,
            axx_mult=axx_mult,
            sa_rows=sa_rows,
            sa_cols=sa_cols,
            src=src,
        )


    
    def build(
        self,
        *,
        op: str,
        use_exact: bool,
        axx_mult: str,
        sa_rows: int = 16,
        sa_cols: int = 16,
        src: Optional[str] = None,
    ):
        self._validate_op(op)
        src_path = self._resolve_source(op, src)
        if not src_path.exists():
            raise FileNotFoundError(
                f"Kernel source not found: {src_path}\nChecked: {self._cpu_kernels_dir}"
            )

        base_name = self._make_module_name(op, axx_mult, use_exact, sa_rows, sa_cols)
        build_dir = Path(self._build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        extra_cflags = list(self._cflags)
        extra_cflags += [
            f"-DAXX_MULT={axx_mult}",
            f"-DSA_ROWS={int(sa_rows)}",
            f"-DSA_COLS={int(sa_cols)}",
        ]
        if use_exact:
            extra_cflags.append("-DUSE_EXACT")

        module = None
        name_to_try = base_name
        salted = False
        try:
            if self.verbose:
                print(f"[SystolicBuilder] Compiling {name_to_try} ...")
                print(f"  src: {src_path}")
                print(f"  build_dir: {build_dir}")
                print(f"  cflags: {' '.join(extra_cflags)}")
                print(f"  ldflags: {' '.join(self._ldflags)}")
                if self._include_paths:
                    print(f"  include_paths: {self._include_paths}")

            module = cpp_load(
                name=name_to_try,
                sources=[str(src_path)],
                extra_cflags=extra_cflags,
                extra_ldflags=self._ldflags,
                extra_include_paths=[str(self._cpu_kernels_dir)] + self._include_paths,
                verbose=self.verbose,
                build_directory=str(build_dir),
                is_python_module=True,
                keep_intermediates=True,
            )
        except ImportError as e:
            # Registry thinks the name exists, but import by unsuffixed name fails.
            if self.verbose:
                print(f"[SystolicBuilder] ImportError for {name_to_try}: {e}")
                print("[SystolicBuilder] Forcing a fresh build with a salted name...")
            salted = True
            name_to_try = f"{base_name}_{uuid.uuid4().hex[:8]}"
            if self.verbose:
                print(f"[SystolicBuilder] Compiling {name_to_try} ...")
                print(f"  src: {src_path}")
                print(f"  build_dir: {build_dir}")
                print(f"  cflags: {' '.join(extra_cflags)}")
                print(f"  ldflags: {' '.join(self._ldflags)}")

            module = cpp_load(
                name=name_to_try,
                sources=[str(src_path)],
                extra_cflags=extra_cflags,
                extra_ldflags=self._ldflags,
                extra_include_paths=[str(self._cpu_kernels_dir)] + self._include_paths,
                verbose=self.verbose,
                build_directory=str(build_dir),
                is_python_module=False,
                keep_intermediates=True,
            )

        # Make sure torch has loaded the ops (for kernels using TORCH_LIBRARY)
        so_real = Path(self._find_shared_object(build_dir, name_to_try))
        if so_real.exists():
            torch.ops.load_library(str(so_real))

        # Create an unsuffixed alias .so so later plain imports by base_name will succeed
        alias_so = build_dir / f"{base_name}.so"
        if so_real != alias_so:
            try:
                if alias_so.exists() or alias_so.is_symlink():
                    alias_so.unlink()
                try:
                    alias_so.symlink_to(so_real)  # best: lightweight
                except Exception:
                    shutil.copy2(so_real, alias_so)  # fallback: duplicate
            except Exception as e:
                if self.verbose:
                    print(f"[SystolicBuilder] alias creation failed: {alias_so} → {so_real} ({e})")

        key = (op, axx_mult, use_exact, sa_rows, sa_cols)
        self._compiled_modules[key] = module
        return module

    def clear_memory_cache(self) -> None:
        self._compiled_modules.clear()

    # --------------------- internal helpers ---------------------

    def _validate_op(self, op: str) -> None:
        if op not in self._default_sources:
            valid = ", ".join(self._default_sources.keys())
            raise ValueError(f"Unsupported op '{op}'. Valid: {valid}")

    def _resolve_source(self, op: str, src: Optional[str]) -> Path:
        if src is not None:
            return Path(src).resolve()
        return self._default_sources[op]

    @staticmethod
    def _make_module_name(
        op: str, axx_mult: str, use_exact: bool, sa_rows: int, sa_cols: int
    ) -> str:
        return f"systolic_{op}_{axx_mult}_exact{str(use_exact)}_r{sa_rows}_c{sa_cols}"

    @staticmethod
    def _find_shared_object(build_dir: Path, name: str) -> str:
        cand = build_dir / f"{name}.so"
        if cand.exists():
            return str(cand)
        # torch may bump internal version (e.g., _v2). Pick the newest match.
        matches = sorted(glob.glob(str(build_dir / f"{name}*.so")))
        return matches[-1] if matches else str(cand)

# ------------------ convenience precompiler ------------------

def precompile_systolic_extensions(
    *,
    axx_mult: str = "mul8s_acc",
    use_exact_variants: Iterable[bool] = (True, False),
    sa_rows: int = 16,
    sa_cols: int = 16,
    ops: Iterable[str] = ("linear", "conv2d"),
    verbose: bool = True,
    build_dir: Optional[str] = None,
    include_paths: Optional[Iterable[str]] = None,
    cflags: Optional[Iterable[str]] = None,
    ldflags: Optional[Iterable[str]] = None,
) -> None:
    builder = SystolicBuilder(
        verbose=verbose,
        build_dir=build_dir,
        include_paths=include_paths,
        cflags=cflags,
        ldflags=ldflags,
    )
    print(f"Pre-compiling systolic extensions for {axx_mult}...")
    for use_exact in use_exact_variants:
        mode = "exact" if use_exact else "approx"
        print(f"  Mode: {mode}")
        for op in ops:
            print(f"    • {op} (r{sa_rows}×c{sa_cols})")
            builder.build(
                op=op,
                use_exact=use_exact,
                axx_mult=axx_mult,
                sa_rows=sa_rows,
                sa_cols=sa_cols,
            )
    print("Pre-compilation complete! Models will now load instantly.")
