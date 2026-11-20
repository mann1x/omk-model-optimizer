#!/bin/bash
# bm898_ladder_chain_kq.sh -- the K-quant lane of the bm-898 K-FAC ladder (owner 2026-11-19 17:39 CET: "before starting
# the K-FAC K-quants complete the OMK CUDA kernels"): runs Q6_K, Q4_K_M, Q5_K_M, Q5_K_S, Q4_K_S, Q3_K_M through
# kfac_ladder_lane.sh / kfac_ladder_twins.sh (model case 20b-base) on the code tree code/omk-wip-kern, whose K-quant +
# legacy CUDA encoders (omk_qad.kquant: Q2_K..Q6_K, Q4_0/Q4_1/Q5_0/Q5_1; bm-901) must first be byte-verified against
# libggml b12386 on this host (KQ_RUN_DONE without FAIL, KQ_EXIT 0, SASS counts 0).
# Runs BESIDE the IQ/Q8_0 lane (bm898_ladder_chain.sh, lane base20) on GPU0 -- disjoint tiers, two rounders fit.
# Waits: (1) the kquant verify of the tree; (2) KFI_DONE of the factors + eval reference chain; (3) .rm_ptq_granted.
# The plain Q4_K_M and Q6_K GGUFs (step 3a, the same llama-quantize b12386 call the twins make) are MOVED into
# ladder/<T>/ptq.gguf: the twins skip the PTQ step ("ptq.gguf exists, kept") and remove it after both scores under
# the owner's marker -- nothing unneeded stays on bs2. Sentinels LK_START / LK_KERNELS_OK / LK_INPUTS / LK_PTQ_MOVED /
# LK_LANE / LK_DONE / LK_FAIL in the chain log + the lane's LN_*.
# Start: KQ_LOG=<verify log> setsid nohup bash .../bm898_ladder_chain_kq.sh > .../ladder_kq.chain.log 2>&1 < /dev/null &
set -uo pipefail
RUNS=/mnt/sdc/omk-model-optimizer/runs; B=$RUNS/bm898_f16
TREE=/mnt/sdc/omk-model-optimizer/code/omk-wip-kern
K=$RUNS/g4_20b_base_kfac; GG=/mnt/sdc/ml/v9a_base/mlk8-noa-sft11/gguf; NAME=gemma-4-20B-v9-agentic-base-mlk8-noa-sft11
KQ_LOG=${KQ_LOG:?path of the verify_kquant_tree log}
KFILOG=$B/kfac_inputs.chain.2.log
TIERS="Q6_K Q4_K_M Q5_K_M Q5_K_S Q4_K_S Q3_K_M"
LOG=$B/ladder_kq.driver.log
say() { echo "[lk $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "LK_FAIL $*"; exit 1; }
[ -s $TREE/omk_qad/kquant.py ] && [ -s $TREE/omk_qad/kernels/kquant.cu ] || die "no kquant code in $TREE"
grep -q '"Q4_K": "kquant"' $TREE/omk_qad/fakequant.py && grep -q '"Q4_0", "Q4_1", "Q5_0", "Q5_1")  # project kernels' $TREE/omk_qad/kfac_round.py || die "tree not wired for kquant"
[ -s $RUNS/kfac_ladder_lane.sh ] && grep -q "20b-base" $RUNS/kfac_ladder_lane.sh || die "runs/kfac_ladder_lane.sh lacks the 20b-base case"
say "LK_START tree $TREE tiers [$TIERS] kquant-log $KQ_LOG"

# ---- 1. the kquant verify of this tree must be green ----------------------------------------------------------------
t=0
until grep -q "KQ_RUN_DONE" $KQ_LOG 2>/dev/null; do [ $t -ge 7200 ] && die "kquant verify not done in 2 h ($KQ_LOG)"; sleep 30; t=$((t + 30)); done
grep -q "KQ_RUN_DONE FAIL" $KQ_LOG && die "kquant verify failed: $(grep -E 'MISMATCH|MISSING|Error' $KQ_LOG | head -n 2 | tr '\n' ' ')"
grep -q "^KQ_EXIT 0" $KQ_LOG || die "kquant verify exit != 0: $(grep KQ_EXIT $KQ_LOG)"
n_so=$(grep -c "^SASS_F2I_U8_F16 " $KQ_LOG); bad=$(grep "^SASS_F2I_U8_F16 " $KQ_LOG | awk '$2 != 0' | wc -l)
[ "$n_so" -ge 1 ] && [ "$bad" -eq 0 ] || die "SASS gate: $n_so .so files, $bad with F2I.U8.F16"
say "LK_KERNELS_OK $(grep 'KQ_VERIFY .* identical over' $KQ_LOG | cut -c1-120)"

# ---- 2. GPU0: the factors + eval reference --------------------------------------------------------------------------
t=0
until grep -qE "KFI_DONE|KFI_FAIL" $KFILOG 2>/dev/null; do [ $t -ge 21600 ] && die "K-FAC inputs chain not done in 6 h"; sleep 60; t=$((t + 60)); done
grep -q KFI_DONE $KFILOG || die "K-FAC inputs chain failed: $(grep KFI_FAIL $KFILOG | tail -n 1 | cut -c1-200)"
say "LK_INPUTS $(grep KFI_DONE $KFILOG | tail -n 1 | cut -c1-200)"
[ -e $K/ladder/.rm_ptq_granted ] || die "no .rm_ptq_granted marker in $K/ladder (owner granted it 2026-11-19 17:39)"

# ---- 3. the plain Q4_K_M / Q6_K become the ptq twins (moved, never copied: nothing unneeded stays) -----------------
for T in Q4_K_M Q6_K; do
  D=$K/ladder/$T; mkdir -p $D
  if [ -s $D/ptq.gguf ]; then say "$T: ladder ptq.gguf exists, plain tier left alone"; continue; fi
  if [ -s $GG/$NAME-$T.gguf ]; then
    mv $GG/$NAME-$T.gguf $D/ptq.gguf && { [ -s $GG/$NAME-$T.gguf.sha256 ] && mv $GG/$NAME-$T.gguf.sha256 $D/ptq.gguf.sha256; }
    say "LK_PTQ_MOVED $T $(stat -c %s $D/ptq.gguf) B from $GG"
  else
    say "$T: no plain tier in $GG, the twins will quantize it"
  fi
done

# ---- 4. the lane -----------------------------------------------------------------------------------------------------
say "LK_LANE base20kq 20b-base (lane log $RUNS/kfac_ladder_lane_base20kq.log)"
OMK_CODE=$TREE TW=$RUNS/kfac_ladder_twins.sh MIN_FREE_G=70 bash $RUNS/kfac_ladder_lane.sh base20kq 20b-base $TIERS
say "LK_DONE $(grep LN_DONE $RUNS/kfac_ladder_lane_base20kq.log | tail -n 1 | cut -c1-160)"
