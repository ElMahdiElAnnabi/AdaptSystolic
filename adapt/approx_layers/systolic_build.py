
# Small helper to ensure exact/approx variants NEVER share a cached module.

from torch.utils.cpp_extension import load
import time


class SystolicBuilder:
    """
    Build/load uniquely-named C++ extensions so exact/approx never collide in the cache.
    Usage:
        builder = SystolicBuilder(sa_rows=16, sa_cols=16, verbose=False)
        ext_lin  = builder.build(op="linear", use_exact=True,  axx_mult="mul8s_acc",
                                 src="/workspace/adapt/adapt/cpu-kernels/axx_linear_systolic.cpp")
        ext_conv = builder.build(op="conv2d", use_exact=False, axx_mult="mul8s_1L2H",
                                 src="/workspace/adapt/adapt/cpu-kernels/axx_conv2d_systolic.cpp")
    """
    def __init__(self, sa_rows: int = 16, sa_cols: int = 16, verbose: bool = False):
        self.sa_rows = int(sa_rows)
        self.sa_cols = int(sa_cols)
        self.verbose = bool(verbose)



    def build(self, *, op: str, use_exact: bool, axx_mult: str, src: str):
        mode = "exact" if use_exact else "approx"
        name = f"{op}_systolic_{mode}_{axx_mult}_{int(time.time()*1000)}"
        cflags = [
            f"-DAXX_MULT={axx_mult}",
            f"-DSA_ROWS={self.sa_rows}",
            f"-DSA_COLS={self.sa_cols}",
            "-O3", "-march=native", "-fopenmp",
            f"-DVARIANT_TAG_{mode.upper()}=1",  # forces a unique build hash per mode
        ]
        if use_exact:
            cflags.append("-DUSE_EXACT")

        #build_dir = f"/tmp/torch_ext_build/{name}"
        #kwargs = {"build_directory": build_dir}
        kwargs={}

        ext = load(
            name=name,
            sources=[src],
            extra_cflags=cflags,
            extra_ldflags=["-lgomp"],
            verbose=self.verbose,
            **kwargs,
        )
        return ext
