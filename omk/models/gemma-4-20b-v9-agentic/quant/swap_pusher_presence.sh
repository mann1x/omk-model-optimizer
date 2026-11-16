#!/bin/bash
# RUNS ON bs2: waits until the ollama pusher is between tiers, swaps in kfac_ollama_bs2.sh.new, clears the 20B
# .published_ollama markers (owner 2026-11-16 23:47: 20B presence_penalty 0.025 -> 0.035, re-push every 20B tag), relaunches.
R=/mnt/sdc/omk-model-optimizer/runs; L=$R/kfac_ollama_bs2.log
for i in $(seq 1 120); do
  last=$(grep -E "KO_TIER |KO_TIER_DONE|KO_FAIL|KO_SKIP" $L | tail -1)
  case "$last" in *KO_TIER\ *) sleep 30 ;; *) break ;; esac
done
P=$(cat $R/kfac_ollama_bs2.pid); kids=$(ps -o pid= --ppid $P | tr '\n' ' '); kill $P $kids 2>/dev/null; sleep 2
kill -0 $P 2>/dev/null && { echo "pusher $P still alive"; exit 1; }
mv $R/kfac_ollama_bs2.sh.new $R/kfac_ollama_bs2.sh || exit 1
for d in $R/g4_20b_kfac/ladder/*/; do [ -e $d/.published_ollama ] && { mv $d/.published_ollama $d/.published_ollama.pres0025; echo "re-push queued: $(basename $d)"; }; done
setsid nohup bash $R/kfac_ollama_bs2.sh > $R/kfac_ollama_bs2.launcher.out 2>&1 < /dev/null & echo $! > $R/kfac_ollama_bs2.pid
echo "SWAP_DONE pusher $(cat $R/kfac_ollama_bs2.pid); last line before swap: $last"
