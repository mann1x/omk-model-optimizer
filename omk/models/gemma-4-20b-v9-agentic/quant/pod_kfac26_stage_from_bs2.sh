#!/bin/bash
# pod_kfac26_stage_from_bs2.sh -- RUNS ON bs2. Stages the 26B K-FAC ladder inputs onto the eval pod (ssh1.vast.ai:29914,
# 2x3090, 206G free) so the 26B tiers can be built there once the Q6_K suite releases the GPUs (owner 2026-11-16 20:42
# "use bs2 for the qad" -> the 26B lane stopped on bs2; 22:29 "what about the pod ladder?"). bs2->pod measured 10 MB/s:
# factors 77G + base 48G ~= 3.5 h serial; the two big streams run in parallel to probe for more aggregate bandwidth.
# Resumable (rsync --partial --inplace). Nothing on bs2 is modified. Sentinels PS_START/PS_SMALL/PS_CODE/PS_FACTORS/
# PS_BASE/PS_FAIL/PS_DONE in runs/pod_kfac26_stage.log.
set -u
KEY=/root/.ssh/id_ed25519_vastpod; H=${POD_HOST:-ssh1.vast.ai}; P=${POD_PORT:-29914}
S="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $P"
R=/mnt/sdc/omk-model-optimizer; L=$R/runs/pod_kfac26_stage.log
C=$R/runs/g4_agentic_iq2_r16k
BASE=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-it-bm889-sft11-sfinal-BF16-eos106.gguf
FACT=$R/runs/g4_agentic_kfac/factors_w512_s2048_g10
D=/workspace/kfac26
say() { echo "[ps $(date -u +%FT%TZ)] $*" | tee -a $L; }
$S root@$H "mkdir -p $D/factors_w512_s2048_g10 $D/cache $D/ladder $D/code" || { say "PS_FAIL pod unreachable"; exit 1; }
say "PS_START -> $H:$P:$D  factors $(du -sh $FACT | cut -f1) base $(du -sh $BASE | cut -f1)"
rsync -a --partial --inplace -e "$S" $C/cache/eval.pt $C/cache/imatrix.gguf $C/step0_gguf_student_eval.py root@$H:$D/cache/ \
  && say "PS_SMALL eval.pt imatrix.gguf step0 ok" || say "PS_FAIL small files"
rsync -a --partial --inplace --exclude '__pycache__' --exclude '.git' -e "$S" $R/code/omk-wip-iq2w/ root@$H:$D/code/omk-wip-iq2w/ \
  && say "PS_CODE omk-wip-iq2w ok" || say "PS_FAIL code"
( t0=$(date +%s); rsync -a --partial --inplace -e "$S" $FACT/ root@$H:$D/factors_w512_s2048_g10/ \
    && say "PS_FACTORS ok in $(( $(date +%s) - t0 )) s" || say "PS_FAIL factors rc=$?" ) &
( t0=$(date +%s); rsync -a --partial --inplace -e "$S" $BASE root@$H:$D/base.gguf \
    && say "PS_BASE ok in $(( $(date +%s) - t0 )) s" || say "PS_FAIL base rc=$?" ) &
for _ in $(seq 1 1000); do
  sleep 300
  jobs -r | grep -q . || break
  say "progress: $($S root@$H "du -sm $D/factors_w512_s2048_g10 $D/base.gguf 2>/dev/null | cut -f1 | tr '\n' ' '") MB (factors base)"
done
wait
want=$(cut -d' ' -f1 $BASE.sha256 2>/dev/null)
if [ ${#want} = 64 ]; then
  got=$($S root@$H "sha256sum $D/base.gguf | cut -d' ' -f1")
  [ "$got" = "$want" ] && say "PS_BASE_SHA ok $got" || say "PS_FAIL base sha pod $got != bs2 $want"
else
  say "no base sidecar on bs2; pod sha $($S root@$H "sha256sum $D/base.gguf | cut -d' ' -f1") (bs2 sha to be computed separately)"
fi
say "PS_DONE $($S root@$H "du -sh $D | cut -f1; df -h /workspace | tail -1")"
