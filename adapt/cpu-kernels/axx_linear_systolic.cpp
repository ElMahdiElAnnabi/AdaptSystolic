#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdint>

// ------- Configure array size (compile-time) -------
#ifndef SA_ROWS
#define SA_ROWS 16
#endif
#ifndef SA_COLS
#define SA_COLS 16
#endif

// ------- LUT include/macros (unchanged) -----------
#define STR2(s) #s
#define STR(s) STR2(s)
#define EXPAND(s) s
#include STR(axx_mults/EXPAND(AXX_MULT).h)

// ------- exact vs LUT multiply (unchanged) --------
static inline int mul_int8(uint8_t a, uint8_t b) {
#ifdef USE_EXACT
    return static_cast<int>(static_cast<int8_t>(a)) * static_cast<int>(static_cast<int8_t>(b));
#else
    return lut[a][b];
#endif
}

// ------- 16x16 systolic-tiling outer-product GEMM --
static at::Tensor systolic_linear_forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.device().is_cpu() && B.device().is_cpu(), "CPU tensors expected");
    TORCH_CHECK(A.dim()==2 && B.dim()==2, "linear: both inputs must be 2D");
    TORCH_CHECK(A.scalar_type()==at::kChar && B.scalar_type()==at::kChar,
                "linear: int8 expected (torch.int8)");

    // Expect A: [M,K], B: [N,K]  → computes C = A @ B^T → [M,N]
    at::Tensor Acontig = A.contiguous();
    at::Tensor Bcontig = B.contiguous();

    const int64_t M = Acontig.size(0);
    const int64_t K = Acontig.size(1);
    TORCH_CHECK(Bcontig.size(1)==K, "K mismatch: A[M,K] vs B[N,K]");
    const int64_t N = Bcontig.size(0);

    at::Tensor C = at::zeros({M, N}, A.options().dtype(at::kInt));

    auto Aacc = Acontig.accessor<int8_t, 2>();   // [M,K]
    auto Bacc = Bcontig.accessor<int8_t, 2>();   // [N,K]
    auto Cacc = C.accessor<int32_t, 2>();        // [M,N]

    constexpr int R = SA_ROWS;
    constexpr int Cc = SA_COLS;

    // Tile the MxN output with an R x Cc systolic array
    for (int64_t m0 = 0; m0 < M; m0 += R) {
        const int64_t mT = std::min<int64_t>(R, M - m0);
        for (int64_t n0 = 0; n0 < N; n0 += Cc) {
            const int64_t nT = std::min<int64_t>(Cc, N - n0);

            // Local PE-accumulators for this tile
            // (same numerics as global, just grouped like a PE array)
            std::vector<int32_t> acc(mT * nT, 0);

            // Stream K (the systolic "time" dimension)
            for (int64_t t = 0; t < K; ++t) {
                // Outer product of A_tile[:, t] and B_tile[:, t]
                for (int64_t i = 0; i < mT; ++i) {
                    const uint8_t a = static_cast<uint8_t>(Aacc[m0 + i][t]); // A[m0+i, t]
                    for (int64_t j = 0; j < nT; ++j) {
                        const uint8_t b = static_cast<uint8_t>(Bacc[n0 + j][t]); // B[n0+j, t]
                        acc[i * nT + j] += mul_int8(a, b);
                    }
                }
            }

            // Write tile back to global C
            for (int64_t i = 0; i < mT; ++i)
                for (int64_t j = 0; j < nT; ++j)
                    Cacc[m0 + i][n0 + j] = acc[i * nT + j];
        }
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &systolic_linear_forward, "systolic_linear_forward (tiled 16x16)");
}


// #include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <cstdint>

// // LUT include
// #define STR2(s) #s
// #define STR(s) STR2(s)
// #define EXPAND(s) s
// #include STR(axx_mults/EXPAND(AXX_MULT).h)

// // exact vs LUT multiply
// static inline int mul_int8(uint8_t a, uint8_t b) {
// #ifdef USE_EXACT
//     return static_cast<int>(static_cast<int8_t>(a)) * static_cast<int>(static_cast<int8_t>(b));
// #else
//     return lut[a][b];
// #endif
// }

// static at::Tensor systolic_linear_forward(at::Tensor A, at::Tensor B) {
//     TORCH_CHECK(A.device().is_cpu() && B.device().is_cpu(), "CPU tensors expected");
//     TORCH_CHECK(A.dim()==2 && B.dim()==2, "linear: both inputs must be 2D");
//     TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar,
//                 "linear: int8 expected (torch.int8)");

//     // make contiguous TENSORS first (lvalues), then take accessors
//     at::Tensor Acontig = A.contiguous();
//     at::Tensor Bcontig = B.contiguous();

//     const int64_t M = Acontig.size(0);
//     const int64_t K = Acontig.size(1);
//     TORCH_CHECK(Bcontig.size(1)==K, "K mismatch: A[M,K] vs B[N,K]");
//     const int64_t N = Bcontig.size(0);

//     at::Tensor C = at::zeros({M, N}, A.options().dtype(at::kInt));

//     auto Aacc = Acontig.accessor<int8_t, 2>();   // [M,K]
//     auto Bacc = Bcontig.accessor<int8_t, 2>();   // [N,K]
//     auto Cacc = C.accessor<int32_t, 2>();        // [M,N]

//     // systolic outer-product sweep over K
//     for (int64_t t = 0; t < K; ++t) {
//         for (int64_t i = 0; i < M; ++i) {
//             const uint8_t a = static_cast<uint8_t>(Aacc[i][t]);
//             for (int64_t j = 0; j < N; ++j) {
//                 const uint8_t b = static_cast<uint8_t>(Bacc[j][t]);
//                 Cacc[i][j] += mul_int8(a, b);
//             }
//         }
//     }
//     return C;
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("forward", &systolic_linear_forward, "systolic_linear_forward");
// }
