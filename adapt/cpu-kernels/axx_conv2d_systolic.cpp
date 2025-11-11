#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

#ifndef SA_ROWS
#define SA_ROWS 16
#endif
#ifndef SA_COLS
#define SA_COLS 16
#endif

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
    std::vector<int64_t> padding) {
    
    TORCH_CHECK(input.device().is_cpu() && weight.device().is_cpu(), "CPU tensors expected");
    TORCH_CHECK(input.dim()==4 && weight.dim()==4, "conv2d: 4D tensors expected");
    TORCH_CHECK(input.scalar_type()==at::kChar && weight.scalar_type()==at::kChar,
                "conv2d: int8 expected (torch.int8)");

    // Input: [N, C, H, W], Weight: [O, C, Kh, Kw]
    at::Tensor input_contig = input.contiguous();
    at::Tensor weight_contig = weight.contiguous();

    const int64_t N = input_contig.size(0);
    const int64_t C = input_contig.size(1);
    const int64_t H = input_contig.size(2);
    const int64_t W = input_contig.size(3);

    const int64_t O  = weight_contig.size(0);
    const int64_t Kh = kernel_size[0];
    const int64_t Kw = kernel_size[1];

    TORCH_CHECK(weight_contig.size(1)==C, "conv2d: channel mismatch");
    TORCH_CHECK(weight_contig.size(2)==Kh && weight_contig.size(3)==Kw, "weight shape mismatch");

    const int64_t Sh = stride[0], Sw = stride[1];
    const int64_t Ph = padding[0], Pw = padding[1];

    const int64_t Ho = (H + 2*Ph - Kh)/Sh + 1;
    const int64_t Wo = (W + 2*Pw - Kw)/Sw + 1;

    // Use im2col to transform input to [N, C*Kh*Kw, Ho*Wo]
    auto X_unf = torch::nn::functional::unfold(
        input_contig.to(torch::kFloat32),  // unfold needs float
        torch::nn::functional::UnfoldFuncOptions({Kh, Kw}).stride({Sh, Sw}).padding({Ph, Pw})
    ).to(torch::kInt8);

    // Reshape weight to [O, C*Kh*Kw] 
    auto W_flat = weight_contig.view({O, C*Kh*Kw});

    // Output: [N, O, Ho*Wo] -> will be reshaped to [N, O, Ho, Wo]
    at::Tensor output = at::zeros({N, O, Ho*Wo}, input.options().dtype(at::kInt));

    auto Xacc = X_unf.accessor<int8_t, 3>();   // [N, C*Kh*Kw, Ho*Wo]
    auto Wacc = W_flat.accessor<int8_t, 2>();  // [O, C*Kh*Kw]
    auto Oacc = output.accessor<int32_t, 3>(); // [N, O, Ho*Wo]

    constexpr int R = SA_ROWS;
    constexpr int Cc = SA_COLS;

    const int64_t K = C * Kh * Kw;
    const int64_t L = Ho * Wo;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t o0 = 0; o0 < O; o0 += R) {
            const int64_t oT = std::min<int64_t>(R, O - o0);
            for (int64_t l0 = 0; l0 < L; l0 += Cc) {
                const int64_t lT = std::min<int64_t>(Cc, L - l0);

                // Local PE-accumulators for this tile
                std::vector<int32_t> acc(oT * lT, 0);

                // Stream K dimension
                for (int64_t t = 0; t < K; ++t) {
                    // Outer product of W_tile[:, t] and X_tile[t, :]
                    for (int64_t i = 0; i < oT; ++i) {
                        const uint8_t w = static_cast<uint8_t>(Wacc[o0 + i][t]);
                        for (int64_t j = 0; j < lT; ++j) {
                            const uint8_t x = static_cast<uint8_t>(Xacc[n][t][l0 + j]);
                            acc[i * lT + j] += mul_int8(x, w);
                        }
                    }
                }

                // Write tile back
                for (int64_t i = 0; i < oT; ++i) {
                    for (int64_t j = 0; j < lT; ++j) {
                        Oacc[n][o0 + i][l0 + j] = acc[i * lT + j];
                    }
                }
            }
        }
    }

    return output.view({N, O, Ho, Wo});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &systolic_conv2d_forward, "systolic_conv2d_forward");
}
