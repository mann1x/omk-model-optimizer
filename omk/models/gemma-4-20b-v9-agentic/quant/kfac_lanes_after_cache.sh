#!/usr/bin/env bash
# kfac_lanes_after_cache.sh — RUNS ON bs2. GPU0 holds the 20B QAD teacher cache (85 GB) until ~00:30 CET, so only one
# rounder fits beside it and no GGUF scoring. This waits for the cache (g4_20b_iq2/cache/state.json status done), then
# starts both ladder lanes with every stage (they score what the build-only lane left unscored, skip finished tiers).
# MIN_FREE_G=170 keeps 150 GB on /mnt/sdc for the QAD train stage. Sentinels LAC_START / LAC_LAUNCHED / LAC_FAIL.
set -u
R=/mnt/sdc/omk-model-optimizer/runs; LOG=$R/kfac_lanes_after_cache.log
say() { echo "[lac $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
say "LAC_START"
for i in $(seq 1 1440); do
  s=$(python3 -c "import json;print(json.load(open('$R/g4_20b_iq2/cache/state.json')).get('status',''))" 2>/dev/null)
  [ "$s" = done ] && break
  grep -q "QAD20_FAIL" $R/g4_20b_iq2/chain.driver.log 2>/dev/null && { say "LAC_FAIL the 20B chain failed before the cache finished: $(grep QAD20_FAIL $R/g4_20b_iq2/chain.driver.log | tail -1 | cut -c1-160)"; break; }
  [ "$i" = 1440 ] && { say "LAC_FAIL cache not done after 12 h"; exit 1; }
  sleep 30
done
say "cache status: ${s:-?}; GPU0 $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
ps -eo args | grep -q "[k]fac_ladder_lane.sh B20" && say "lane B20 (build) still running; the full-stage lane waits for it" && while ps -eo args | grep -q "[k]fac_ladder_lane.sh B20"; do sleep 60; done
MIN_FREE_G=170 setsid nohup bash $R/kfac_ladder_lane.sh B20 20b IQ4_XS Q5_K_S Q4_K_S Q5_K_M IQ4_NL Q3_K_M Q8_0 IQ3_XXS IQ3_M IQ3_XS IQ2_M IQ2_S IQ2_XS >> $R/kfac_ladder_lane_B20.launcher.out 2>&1 < /dev/null &
sleep 2
MIN_FREE_G=170 setsid nohup bash $R/kfac_ladder_lane.sh C26 26b Q5_K_S Q4_K_S Q5_K_M Q6_K IQ4_NL Q3_K_M Q8_0 IQ3_XXS IQ3_M IQ3_XS IQ2_M IQ2_S IQ2_XS >> $R/kfac_ladder_lane_C26.launcher.out 2>&1 < /dev/null &
sleep 5; say "LAC_LAUNCHED B20 + C26 (all stages, min free 170G): $(ps -eo pid,args | grep '[k]fac_ladder_lane.sh' | grep -v 'bash -c' | awk '{print $1}' | tr '\n' ' ')"
