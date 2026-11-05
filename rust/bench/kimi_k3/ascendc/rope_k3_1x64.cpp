/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "kernel_operator.h"

#ifndef TILE_OUT_FILL
#define TILE_OUT_FILL 0
#endif

class Kernel_k3_rope_64 {
public:
    __aicore__ inline Kernel_k3_rope_64() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        pipe.InitBuffer(tbuf0, 256);
        pipe.InitBuffer(tbuf1, 256);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        int32_t __v3;
        __v0 = 0;
        __v1 = 0;
        __v2 = 1;
        __v3 = 64;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (64));
        AscendC::PipeBarrier<PIPE_ALL>();
        // TILE_SUBSTITUTION: intrinsic=__tile_rope_f32; replacement=identity-copy; reason=missing-cos-sin-operands
        { // tile_rope (identity pending cos/sin LUT signature)
          int __rp_n = ((__v2) * (__v3));
          for (int __i = 0; __i < __rp_n; ++__i)
            buf1.SetValue(__i, buf0.GetValue(__i));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v2) * (__v3)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm1Tensor, buf1, ((__v2) * (__v3))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm1Tensor, buf1, __sp); } }
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf1;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
};

extern "C" __global__ __aicore__ void k3_rope_64(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_k3_rope_64 op;
    op.Init(gm0, gm1);
    op.Process();
}


extern "C" void k3_rope_64_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1) {
    k3_rope_64<<<blockDim, nullptr, stream>>>(gm0, gm1);
}
