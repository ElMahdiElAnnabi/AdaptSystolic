#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// LUT include (same mechanism)
#define STR2(s) #s
#define STR(s) STR2(s)
#define EXPAND(s) s
#include STR(axx_mults/EXPAND(AXX_MULT).h)

static inline int mul_int8(uint8_t a, uint8_t b) {
#ifdef USE_EXACT
    return static_cast<int>(static_cast<int8_t>(a)) * static_cast<int>(static_cast<int8_t>(b));
#else
    return lut[a][b];
#endif
}

// input:  [N, C, H, W] int8
// weight: [O, C, Kh, Kw] int8
// kernel_size, stride, padding: each length-2 vector {h, w}
// returns: [N, O, Ho, Wo] int32
static torch::Tensor systolic_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding)
{
    TORCH_CHECK(input.dtype()==torch::kInt8 && weight.dtype()==torch::kInt8, "conv2d: int8 expected");
    TORCH_CHECK(input.dim()==4 && weight.dim()==4, "conv2d: 4D tensors expected");

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);

    const int64_t O  = weight.size(0);
    const int64_t Kh = weight.size(2);
    const int64_t Kw = weight.size(3);

    TORCH_CHECK((int64_t)kernel_size[0]==Kh && (int64_t)kernel_size[1]==Kw, "kernel_size mismatch");
    TORCH_CHECK(weight.size(1)==C, "conv2d: channel mismatch");

    const int64_t Sh = stride[0], Sw = stride[1];
    const int64_t Ph = padding[0], Pw = padding[1];

    const int64_t Ho = (H + 2*Ph - Kh)/Sh + 1;
    const int64_t Wo = (W + 2*Pw - Kw)/Sw + 1;

    // im2col: X_unf: [N, C*Kh*Kw, Ho*Wo]
    auto X_unf = torch::nn::functional::unfold(
        input.to(torch::kFloat32),  // unfold needs float; we’ll cast back
        torch::nn::functional::UnfoldFuncOptions({Kh, Kw}).stride({Sh, Sw}).padding({Ph, Pw})
    ).to(torch::kInt8);

    // W_flat: [O, C*Kh*Kw]
    auto W_flat = weight.contiguous().view({O, C*Kh*Kw});

    auto Y = torch::zeros({N, O, Ho*Wo}, torch::dtype(torch::kInt32).device(input.device()));

    auto Wacc = W_flat.accessor<int8_t,2>();
    auto Xacc = X_unf.accessor<int8_t,3>();
    auto Yacc = Y.accessor<int32_t,3>();

    const int64_t K = C*Kh*Kw;
    const int64_t L = Ho*Wo;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t t = 0; t < K; ++t) {
            for (int64_t i = 0; i < O; ++i) {
                const uint8_t w = static_cast<uint8_t>(Wacc[i][t]);
                for (int64_t j = 0; j < L; ++j) {
                    const uint8_t x = static_cast<uint8_t>(Xacc[n][t][j]);
                    Yacc[n][i][j] += mul_int8(x, w);
                }
            }
        }
    }
    return Y.view({N, O, Ho, Wo});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &systolic_conv2d_forward, "systolic_conv2d_forward");
}
