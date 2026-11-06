// Engine-signature shims: the MERGED Rust kernels exposed under the names and signatures the
// serving engine binds. `device.rs` looks up C++-MANGLED device symbols, so these are plain C++
// `__global__` functions (no `extern "C"`), each instantiating the generated kernel class.
//
// This must be concatenated AFTER the kernel classes, which is what merge_kernels.py does.
//
// Covered here: the slots whose socket already matches a Rust kernel's argument list exactly.
//   ds4_rmsmul_{4096,1024,512}(float*, float*, float*)   3 args, rms x gamma
//   ds4_hc_norm(float*, float*)                          2 args, no gamma
//   ds4_rope_halves(float*, float*, float*, float*)      4 args
// NOT covered, and deliberately: the four `ds4_q8mv_*` sockets are 4-argument Q8_0 with a
// companion plane, and `ds4_hc_mv`/`ds4_mx4pk_*` are packed formats -- the w8a8 arms take five
// arguments, so exposing them under those names would be the over-credit iteration 51 removed.

__global__ __aicore__ void ds4_rmsmul_4096(__gm__ float *x, __gm__ float *g, __gm__ float *o) {
    Kernel_engine_rmsmul_1_4096 op;
    op.Init((GM_ADDR)x, (GM_ADDR)g, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_rmsmul_1024(__gm__ float *x, __gm__ float *g, __gm__ float *o) {
    Kernel_engine_rmsmul_1_1024 op;
    op.Init((GM_ADDR)x, (GM_ADDR)g, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_rmsmul_512(__gm__ float *x, __gm__ float *g, __gm__ float *o) {
    Kernel_engine_rmsmul_1_512 op;
    op.Init((GM_ADDR)x, (GM_ADDR)g, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_hc_norm(__gm__ float *x, __gm__ float *o) {
    Kernel_engine_rms_norm_1_16384 op;
    op.Init((GM_ADDR)x, (GM_ADDR)o);
    op.Process();
}

__global__ __aicore__ void ds4_rope_halves(__gm__ float *x, __gm__ float *cs, __gm__ float *sn, __gm__ float *o) {
    Kernel_rope_halves_engine_1x64 op;
    op.Init((GM_ADDR)x, (GM_ADDR)cs, (GM_ADDR)sn, (GM_ADDR)o);
    op.Process();
}
