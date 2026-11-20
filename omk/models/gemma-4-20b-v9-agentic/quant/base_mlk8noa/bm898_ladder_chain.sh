#!/bin/bash
# bm898_ladder_chain.sh -- bm-898 step 3e (owner 2026-11-19 17:25 CET "create the quants of 20b v9-agentic with the new
# CUDA kernels if possible"): the K-FAC ladder of gemma-4-20B-v9-agentic-base-mlk8-noa-sft11 on bs2 GPU0, through
# kfac_ladder_lane.sh / kfac_ladder_twins.sh (model case 20b-base) with the code tree code/omk-wip-kern, whose CUDA
# encoders (IQ4_XS/NL, IQ3_XXS/S, IQ2_XXS/XS/S, Q8_0 -- kfac_round KERNEL_TYPES) are byte-identical to libggml; the
# K-quants (Q3_K..Q6_K) and the attention/embd tensors still go through libggml on the CPU (bm-901 ports pending).
# Order (owner 17:39 CET: K-quants only after the bm-901 CUDA ports): Q8_0, IQ4_XS, IQ4_NL, IQ3_XXS, IQ3_XS, IQ3_M,
# IQ2_M, IQ2_S, IQ2_XS. Q6_K/Q5_K/Q4_K/Q3_K tiers follow in a second lane once the K-quant kernels are verified.
# Waits: (1) the kernel verify of the tree (KT_RUN_DONE, both exits 0, SASS counts 0) -- else LC_FAIL;
#        (2) KFI_DONE of the factors + eval reference chain (GPU0 then free).
# ptq.gguf twins are removed after both scores (.rm_ptq_granted, owner 17:39); the HF publisher uploads each kfac.gguf
# and removes the bs2 copy after the hub sha matches; the lane waits at MIN_FREE_G.
# Sentinels LC_START / LC_KERNELS_OK / LC_INPUTS <state> / LC_LANE / LC_DONE / LC_FAIL in the chain log + the lane's LN_*.
# Start: setsid nohup bash /mnt/sdc/omk-model-optimizer/runs/bm898_f16/bm898_ladder_chain.sh > /mnt/sdc/omk-model-optimizer/runs/bm898_f16/ladder.chain.log 2>&1 < /dev/null &
set -uo pipefail
RUNS=/mnt/sdc/omk-model-optimizer/runs; B=$RUNS/bm898_f16
TREE=/mnt/sdc/omk-model-optimizer/code/omk-wip-kern
K=$RUNS/g4_20b_base_kfac; GG=/mnt/sdc/ml/v9a_base/mlk8-noa-sft11/gguf; NAME=gemma-4-20B-v9-agentic-base-mlk8-noa-sft11
KT_LOG=${KT_LOG:?path of the verify_kern_tree log}
KFILOG=$B/kfac_inputs.chain.2.log   # factors + eval reference (bm898_kfac_inputs.sh)
TIERS="Q8_0 IQ4_XS IQ4_NL IQ3_XXS IQ3_XS IQ3_M IQ2_M IQ2_S IQ2_XS"   # owner 17:39: K-quant tiers (Q3_K..Q6_K) only after the bm-901 CUDA ports
LOG=$B/ladder.driver.log
say() { echo "[lc $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "LC_FAIL $*"; exit 1; }
[ -s $TREE/omk_qad/kfac_round.py ] && [ -s $TREE/omk_qad/kernels/q8.cu ] || die "no kernel code tree at $TREE"
[ -s $RUNS/kfac_ladder_twins.sh ] && grep -q "20b-base" $RUNS/kfac_ladder_twins.sh || die "runs/kfac_ladder_twins.sh lacks the 20b-base case"
[ -s $RUNS/kfac_ladder_lane.sh ] && grep -q "20b-base" $RUNS/kfac_ladder_lane.sh || die "runs/kfac_ladder_lane.sh lacks the 20b-base case"
say "LC_START tree $TREE tiers [$TIERS] kernels-log $KT_LOG"

# ---- 1. the kernel verify of this tree must be green --------------------------------------------------------------
t=0
until grep -q "KT_RUN_DONE" $KT_LOG 2>/dev/null; do [ $t -ge 3600 ] && die "kernel verify not done in 60 min ($KT_LOG)"; sleep 30; t=$((t + 30)); done
grep -q "KT_RUN_DONE FAIL" $KT_LOG && die "kernel verify preflight failed: $(grep MISSING $KT_LOG | head -n 1)"
grep -q "^KT_IQ2_EXIT 0" $KT_LOG || die "IQ2 verify exit != 0: $(grep KT_IQ2_EXIT $KT_LOG)"
grep -q "^KT_Q8_EXIT 0" $KT_LOG || die "Q8 verify exit != 0: $(grep KT_Q8_EXIT $KT_LOG)"
n_so=$(grep -c "^SASS_F2I_U8_F16 " $KT_LOG); bad=$(grep "^SASS_F2I_U8_F16 " $KT_LOG | awk '$2 != 0' | wc -l)
[ "$n_so" -ge 2 ] && [ "$bad" -eq 0 ] || die "SASS gate: $n_so .so files, $bad with F2I.U8.F16"
say "LC_KERNELS_OK $(grep -c IQ2_VERIFY $KT_LOG) IQ2 + $(grep -c Q8_VERIFY $KT_LOG) Q8 verify lines, $n_so .so SASS-clean"

# ---- 2. GPU0: wait for the factors + eval reference ------------------------------------------------------------------
t=0
until grep -qE "KFI_DONE|KFI_FAIL" $KFILOG 2>/dev/null; do [ $t -ge 21600 ] && die "K-FAC inputs chain not done in 6 h"; sleep 60; t=$((t + 60)); done
grep -q KFI_DONE $KFILOG || die "K-FAC inputs chain failed: $(grep KFI_FAIL $KFILOG | tail -n 1 | cut -c1-200)"
say "LC_INPUTS $(grep KFI_DONE $KFILOG | tail -n 1 | cut -c1-200)"
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits); [ "${used:-99999}" -lt 2000 ] || die "GPU0 not free ($used MiB)"

# ---- 3. ptq twins are removed after scoring (owner 17:39: .rm_ptq_granted marker set in $K/ladder) -------------------
[ -e $K/ladder/.rm_ptq_granted ] || die "no .rm_ptq_granted marker in $K/ladder (owner granted it 2026-11-19 17:39)"

# ---- 4. the lane ------------------------------------------------------------------------------------------------------
say "LC_LANE base20 20b-base (lane log $RUNS/kfac_ladder_lane_base20.log)"
OMK_CODE=$TREE TW=$RUNS/kfac_ladder_twins.sh MIN_FREE_G=70 bash $RUNS/kfac_ladder_lane.sh base20 20b-base $TIERS
say "LC_DONE $(grep LN_DONE $RUNS/kfac_ladder_lane_base20.log | tail -n 1 | cut -c1-160)"
