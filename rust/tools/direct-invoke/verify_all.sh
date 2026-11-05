#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Re-run every kernel this tree has verified on an Ascend part, and report which
# still pass.
#
#   ./verify_all.sh <bench-root>
#
# <bench-root> is a checkout's `rust/bench` (the corpus of `.cpp` artefacts).
# Exits non-zero if any kernel fails, so a claim cannot outlive its evidence.
#
# The table below is the evidence list, not a convenience list: each row records
# the launcher, extent, arity, oracle and tolerance that the run used, so a reader
# can see what "verified" meant for each kernel rather than trusting a percentage.
#
# It listed SEVEN kernels while the campaign had twenty-four -- it would have
# printed "all kernels passed" over a third of the evidence. Regenerate this table
# from the records rather than by hand, and check the count against the report.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench="${1:?usage: verify_all.sh <bench-root>}"

# kernel                                    arch        launcher                    n     arity oracle eps
# kernel                                    arch        launcher                    n     arity oracle eps     tol
read -r -d '' ROWS <<'ROWS'
deepseek_v41/ascendc/residual_add_v41_1x5120         dav-2201 v41_residual_add_do          5120 3   0 1e-5f    1e-6
glm_5_3/ascendc/residual_add_glm53_1x4096            dav-2201 glm53_residual_add_do        4096 3   0 1e-5f    1e-6
kimi_k3/ascendc/residual_add_k3_1x7168               dav-2201 k3_residual_add_do           7168 3   0 1e-5f    1e-6
deepseek_v41/ascendc/rmsnorm_v41_1x5120_decode       dav-2201 bm_rmsnorm_1_5120_do         5120 2   1 1e-20f   1e-6
deepseek_v41/ascendc/rmsnorm_v41_8x1280_quarter      dav-2201 bm_rmsnorm_8_1280_do         1280 2   1 1e-20f   1e-6
glm_5_3/ascendc/rmsnorm_glm53_1x4096                 dav-2201 glm53_rmsnorm_1_4096_do      4096 2   1 1e-5f    1e-6
kimi_k3/ascendc/rmsnorm_k3_1x7168                    dav-2201 k3_rmsnorm_1_7168_do         7168 2   1 1e-5f    1e-6
deepseek_v41/ascendc/swiglu_v41_1x2304               dav-2201 bm_swiglu_1_2304_do          2304 3   3 1e-5f    1e-6
glm_5_3/ascendc/shexp_gate_up_silu_glm53_1x2048      dav-2201 glm53_shexp_swiglu_do        2048 4   8 1e-5f    1e-6
glm_5_3/ascendc/routed_gate_up_glm53_8x2048       dav-2201 glm53_routed_gate_up_2048_do   16384 4  8 1e-5f 1e-5
glm_5_3/ascendc/routed_down_glm53_1x2048          dav-2201 glm53_routed_down_2048_do        2048 3  2 1e-5f 1e-5
engine/ascendc/rmsnorm_engine_1x512                  dav-2201 engine_rmsnorm_1_512_do          512 2  1 1e-5f 1e-6
engine/ascendc/rmsnorm_engine_1x1024                 dav-2201 engine_rmsnorm_1_1024_do        1024 2  1 1e-5f 1e-6
kimi_k3/ascendc/kda_delta_k3_1x128                   dav-2201 k3_kda_delta_128_do           128 5   4 1e-5f    1e-6
glm_5_3/ascendc/kda_delta_glm53_1x128                dav-2201 glm53_kda_delta_128_do        128 5   4 1e-5f    1e-6
glm_5_3/ascendc/kda_conv_glm53_1x4096_k4             dav-2201 glm53_kda_conv_4096_do       4096 9  11 1e-5f    1e-5
kimi_k3/ascendc/kda_conv_k3_1x7168_k4                dav-2201 k3_kda_conv_7168_do          7168 9  11 1e-5f    1e-5
kimi_k3/ascendc/shexp_gate_up_situ_k3_1x3072         dav-2201 k3_shexp_situ_do             3072 3  10 1e-5f    1e-5
kimi_k3/ascendc/routed_gate_up_situ_k3_1x3072        dav-2201 k3_routed_situ_do            3072 3  10 1e-5f    1e-5
kimi_k3/ascendc/dense_gate_up_situ_k3_1x33792        dav-2201 k3_dense_situ_do            33792 3  10 1e-5f    1e-5
glm_5_3/ascendc/proj_q_a_glm53_1x1536                dav-2201 glm53_proj_q_a_do            1536 3   2 1e-5f    1e-5
glm_5_3/ascendc/proj_output_glm53_1x4096             dav-2201 glm53_proj_output_do         4096 3   2 1e-5f    1e-5
glm_5_3/ascendc/proj_kv_a_mqa_glm53_1x512            dav-2201 glm53_proj_kv_a_mqa_do        512 3   2 1e-5f    1e-5
glm_5_3/ascendc/proj_shexp_down_glm53_1x4096         dav-2201 glm53_proj_shexp_down_do     4096 3   2 1e-5f    1e-5
glm_5_3/ascendc/proj_dense_down_glm53_1x4096         dav-2201 glm53_proj_dense_down_do     4096 3   2 1e-5f    1e-5
kimi_k3/ascendc/proj_q_a_k3_1x1536                   dav-2201 k3_proj_q_a_do               1536 3   2 1e-5f    1e-5
kimi_k3/ascendc/proj_kv_a_mqa_k3_1x512               dav-2201 k3_proj_kv_a_mqa_do           512 3   2 1e-5f    1e-5
kimi_k3/ascendc/proj_output_k3_1x7168                dav-2201 k3_proj_output_do            7168 3   2 1e-5f    1e-5
ROWS

printf 'host: %s\n' "$(hostname)"
if command -v npu-smi >/dev/null 2>&1; then
    # A row of the product table looks like `| 0     910B2   | OK ...`.
    printf 'part: %s\n' "$(npu-smi info 2>/dev/null | awk -F'|' '{n=split($2,f," "); for(i=1;i<=n;i++) if (f[i] ~ /^9[0-9]+[A-Z0-9]/) {print f[i]; exit}}')"
fi
[ -d "${bench}" ] || { echo "no bench at ${bench}" >&2; exit 2; }
printf 'cann: %s\n' "${ASCEND_HOME_PATH:-unset}"
printf 'ptoas: %s\n\n' "$(ptoas --version 2>/dev/null | head -1 || echo absent)"

printf '%-40s %-8s %s\n' kernel elements result
failed=0
while read -r kernel arch launcher n arity oracle eps tol; do
    [ -z "${kernel}" ] && continue
    out="$(bash "${here}/setup.sh" "${bench}/${kernel}.cpp" \
        "${arch}" "${launcher}" "${n}" "${arity}" "${oracle}" "${eps}" "${tol}" 2>/dev/null | tail -2 | tr '\n' ' ')"
    case "${out}" in
        *PASS*) verdict="PASS  ${out}" ;;
        *)      verdict="FAIL  ${out}"; failed=$((failed + 1)) ;;
    esac
    printf '%-40s %-8s %s\n' "$(basename "${kernel}")" "${n}" "${verdict}"
done <<< "${ROWS}"

printf '\n%s\n' "$([ "${failed}" -eq 0 ] && echo "all kernels passed" || echo "${failed} kernel(s) failed")"
exit $([ "${failed}" -eq 0 ] && echo 0 || echo 1)
