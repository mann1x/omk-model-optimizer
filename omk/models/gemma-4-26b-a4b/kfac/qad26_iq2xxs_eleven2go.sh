#!/bin/bash
# bm-897 (owner 2026-11-16 23:00 CET: "add the 26B IQ2_XXS QAD s600 to eleven2go"). Runs on solidpc. Adds a NEW xollama
# tag on eleven2go (Windows, host .161, xollama :22434):
#   mannix/gemma4-26b-agentic:mtp-vision-IQ2_XXS <- the seeded-QAD export qadE6_kfacseed_s600.gguf (sha 7c9dd585, 9.99 GB,
#   published on HF as gemma-4-26B-A4B-agentic-IQ2_XXS.gguf), pulled from bs2 into google/kfac_ladders/26b_agentic/IQ2_XXS/
#   (sha-verified) first. Modelfile = the mtp-vision-Q4_K_M tag's modelfile (/api/show) with FROM/DRAFT/projector rewritten,
#   so every parameter matches the existing 26B tags. The older arm-E tags (mtp-vision-IQ2-kfac-e, kfac-iq2xxs-e) are kept.
# Same mechanics as kfac_q4km_eleven2go.sh (.ps1 helpers, no inline PowerShell: bug-948/950).
# Sentinels KQE_START / KQE_STEP_OK / KQE_TAG_DONE / KQE_FAIL / KQE_DONE in google/kfac_ladders/eleven2go_iq2xxs.log.
set -uo pipefail
B=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
A=$B/google/kfac_ladders
LOG=$A/eleven2go_iq2xxs.log
XO='C:\Users\ManniX\AppData\Local\Programs\xOllama\xollama.exe'
DRAFT='C:\Users\ManniX\.ollama\models\blobs\sha256-5963a2edfb516fbdff499b1d4b7f8158a7a811af7196eaae00be2119188e422e'
PROJ26='C:\Users\ManniX\.ollama\models\blobs\sha256-2aaf3422ef92970143c037d8865513e18031fc9f14595819d972a302b424853e'
API=http://127.0.0.1:22434
say() { echo "[kqe $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die() { say "KQE_FAIL $*"; claude-hooks-res release eleven2go >/dev/null 2>&1; exit 1; }
E2G() { ssh -o ConnectTimeout=25 eleven2go "$@" < /dev/null; }
runps() { E2G "powershell -NoProfile -ExecutionPolicy Bypass -File $1" | tr -d '\r'; }   # $1 = Windows path of a staged .ps1

# one tag: $1 which $2 NEW tag $3 source gguf (sidecar <gguf>.sha256 beside it) $4 stage dir $5 gguf name $6 experts $7 projector blob $8 template tag (its modelfile is reused)
one() {
  local W=$1 TAG=$2 SRC=$3 STG=$4 NAME=$5 EXP=$6 PROJ=$7 TPL=$8
  local S="C:\\ollama_stage\\$STG" SU="C:/ollama_stage/$STG"
  local D=$A/eleven2go_$STG; mkdir -p $D
  [ -s $SRC ] || die "$W: no $SRC"
  local want got; want=$(cut -d' ' -f1 $3.sha256); got=$(sha256sum $SRC | cut -d' ' -f1)
  [ "$want" = "$got" ] || die "$W: archive sha $got != sidecar $want"
  [ ${#got} = 64 ] || die "$W: bad sha"
  say "$W: source $SRC sha $got ($(du -h $SRC | cut -f1))"
  # --- the .ps1 helpers for this tag
  for t in tpl new; do [ $t = tpl ] && st=$TPL || st=$TAG; cat > $D/show_$t.ps1 <<EOF
\$r = Invoke-RestMethod -Uri $API/api/show -Method Post -Body (@{ model = '$st' } | ConvertTo-Json) -ContentType application/json
"renderer=" + \$r.renderer + " parser=" + \$r.parser + " experts=" + \$r.model_info.'gemma4.expert_count' + " quant=" + \$r.details.quantization_level + " from_count=" + @((\$r.modelfile -split "\`n") | Where-Object { \$_ -like 'FROM *' }).Count + " draft=" + @((\$r.modelfile -split "\`n") | Where-Object { \$_ -like 'DRAFT *' }).Count + " imatrix_file=" + \$r.model_info.'quantize.imatrix.file' + " imatrix_chunks=" + \$r.model_info.'quantize.imatrix.chunks_count'
"---MODELFILE---"
\$r.modelfile
EOF
  done
  cat > $D/free.ps1 <<EOF
"CFREE " + [math]::Round((Get-PSDrive C).Free/1GB,1)
"LOADED " + @((Invoke-RestMethod -Uri $API/api/ps).models).Count
foreach (\$m in (Invoke-RestMethod -Uri $API/api/ps).models) { "LOADEDNAME " + \$m.name }
EOF
  cat > $D/unload.ps1 <<EOF
Invoke-RestMethod -Uri $API/api/generate -Method Post -Body (@{ model = '$TAG'; keep_alive = 0 } | ConvertTo-Json) -ContentType application/json | Out-Null
Start-Sleep 3
"LOADED " + @((Invoke-RestMethod -Uri $API/api/ps).models).Count
EOF
  cat > $D/smoke.ps1 <<EOF
\$b = @{ model = '$TAG'; messages = @(@{ role = 'user'; content = 'Reply with the single word OK.' }); stream = \$false; keep_alive = 0; options = @{ num_predict = 64 } } | ConvertTo-Json -Depth 5
\$r = Invoke-RestMethod -Uri $API/api/chat -Method Post -Body \$b -ContentType application/json -TimeoutSec 900
\$a = [string]\$r.message.content; if (\$a.Length -gt 80) { \$a = \$a.Substring(0, 80) }
"SMOKE eval_count=" + \$r.eval_count + " tok_s=" + [math]::Round(\$r.eval_count / ([double]\$r.eval_duration / 1e9), 1) + " answer=" + (\$a -replace "\`n", " ")
EOF
  printf '(Get-FileHash %s\\%s -Algorithm SHA256).Hash.ToLower()\r\n' "$S" "$NAME" > $D/sha.ps1
  printf '@echo off\r\ncd /d %s\r\nset XOLLAMA_HOST=127.0.0.1:22434\r\n"%s" create %s -f Modelfile > create.log 2>&1\r\necho CREATE_RC=%%ERRORLEVEL%% >> create.log\r\n' "$S" "$XO" "$TAG" > $D/create.cmd
  E2G "cmd /c if not exist $S mkdir $S" || die "$W: mkdir $S"
  E2G "cmd /c if exist $S\\$NAME echo EXISTS" | grep -q EXISTS && die "$W: $S\\$NAME already staged, refusing to overwrite"
  scp -q $D/show_tpl.ps1 $D/show_new.ps1 $D/free.ps1 $D/unload.ps1 $D/smoke.ps1 $D/sha.ps1 $D/create.cmd "eleven2go:$SU/" < /dev/null || die "$W: scp helpers"
  local fr; fr=$(runps "$S\\free.ps1"); echo "$fr" | tee -a $LOG
  local free; free=$(echo "$fr" | grep CFREE | tr -dc 0-9.)
  awk -v f="$free" 'BEGIN{exit !(f+0 >= 60)}' || die "$W: C: only ${free} GB free (need GGUF + blob copy)"
  # --- Modelfile: the tag's current modelfile with FROM/DRAFT/projector rewritten; every PARAMETER/RENDERER/PARSER kept
  local cur; cur=$(runps "$S\\show_tpl.ps1" | sed -n '/^---MODELFILE---$/,$p' | sed 1d)
  echo "$cur" | grep -q "^RENDERER" || die "$W: /api/show gave no modelfile for template $TPL: $(echo "$cur" | head -3)"
  {
    echo "FROM $S\\$NAME"
    echo "DRAFT $DRAFT"
    [ -n "$PROJ" ] && echo "FROM $PROJ"
    # everything except the comments, the FROM/DRAFT lines and the multi-line think_budget_message block ...
    echo "$cur" | awk '/^PARAMETER think_budget_message/{skip=1} skip && /^"[[:space:]]*$/{skip=0; next} skip{next} /^(RENDERER|PARSER|PARAMETER|TEMPLATE|SYSTEM) /{print}'
    # ... which is re-emitted verbatim (from its PARAMETER line to the closing quote line)
    echo "$cur" | awk '/^PARAMETER think_budget_message/{p=1} p{print} p && /^"[[:space:]]*$/{exit}'
  } > $D/Modelfile
  grep -q "^PARAMETER think_budget high" $D/Modelfile || die "$W: Modelfile lost think_budget: $(cat $D/Modelfile)"
  grep -q "^PARAMETER presence_penalty 0.025" $D/Modelfile || die "$W: Modelfile lost presence_penalty"
  grep -q "^PARAMETER draft_num_predict 2" $D/Modelfile || die "$W: Modelfile lost draft_num_predict"
  [ "$(grep -c "think_budget_message" $D/Modelfile)" = 1 ] || die "$W: think_budget_message count != 1"
  [ "$(grep -c "^FROM " $D/Modelfile)" = $([ -n "$PROJ" ] && echo 2 || echo 1) ] || die "$W: FROM count wrong"
  say "$W: Modelfile:"; sed 's/^/    /' $D/Modelfile | tee -a $LOG
  say "$W: scp $NAME -> eleven2go $S"
  local t0=$(date +%s)
  scp -q $SRC "eleven2go:$SU/$NAME" < /dev/null || die "$W: scp gguf"
  scp -q $D/Modelfile "eleven2go:$SU/" < /dev/null || die "$W: scp Modelfile"
  say "$W: copied in $(( $(date +%s) - t0 )) s"
  local rs; rs=$(runps "$S\\sha.ps1" | tr -dc 'a-f0-9')
  [ "$rs" = "$got" ] || die "$W: eleven2go sha '$rs' != solidpc $got"
  say "KQE_STEP_OK $TAG stage sha $got"
  if runps "$S\\free.ps1" | grep -q "LOADEDNAME $TAG"; then say "$W: unloading $TAG"; runps "$S\\unload.ps1" | tee -a $LOG; fi
  say "$W: xollama create $TAG (new tag)"
  E2G "cmd /c $S\\create.cmd"
  local cl; cl=$(E2G "cmd /c type $S\\create.log" | tr -d '\r' | tail -6); echo "$cl" | tee -a $LOG
  echo "$cl" | grep -q "CREATE_RC=0" || die "$W: create rc: $cl"
  local show; show=$(runps "$S\\show_new.ps1" | sed -n '1p')
  say "$W: show: $show"
  echo "$show" | grep -q "experts=$EXP" || die "$W: expert_count != $EXP"
  echo "$show" | grep -qE "quant=IQ2_(M|XXS)" || die "$W: quant not IQ2_M/IQ2_XXS (ftype IQ2_M with IQ2_XXS gate_up)"
  echo "$show" | grep -q "draft=1" || die "$W: drafter missing"
  if [ -n "$PROJ" ]; then echo "$show" | grep -q "from_count=2" || die "$W: projector missing"; fi
  say "KQE_STEP_OK $TAG create"
  local loaded; loaded=$(runps "$S\\free.ps1" | grep "^LOADED " | tr -dc 0-9)
  if [ "${loaded:-1}" = "0" ]; then
    local sm; sm=$(runps "$S\\smoke.ps1"); say "$W: $sm"
    echo "$sm" | grep -q "SMOKE eval_count=[1-9]" || die "$W: smoke produced no tokens"
  else
    say "$W: smoke skipped ($loaded model(s) loaded by someone else)"
  fi
  say "KQE_TAG_DONE $TAG <- $NAME sha $got (staged at $S, kept); template $TPL"
}

say "KQE_START"
SRCDIR=$A/26b_agentic/IQ2_XXS; mkdir -p $SRCDIR
BS2D=/mnt/sdc/omk-model-optimizer/runs/g4_agentic_kfac/ladder/IQ2_XXS
if [ ! -s $SRCDIR/qad.gguf ] || [ "$(sha256sum $SRCDIR/qad.gguf | cut -d' ' -f1)" != "7c9dd585582d309ddf5b25474ddc7949bdc4dbfc4383fa289aef054a50729ba8" ]; then
  say "pulling the QAD s600 export from bs2 (dereferenced symlink) + sidecars"
  rsync -aL --partial --inplace --include='qad.gguf' --include='*.sha256' --include='*.json' --include='*.log' --include='qad.done' --exclude='*' \
        bs2:$BS2D/ $SRCDIR/ >> $LOG 2>&1 || die "rsync from bs2"
fi
got=$(sha256sum $SRCDIR/qad.gguf | cut -d' ' -f1); [ "$got" = "7c9dd585582d309ddf5b25474ddc7949bdc4dbfc4383fa289aef054a50729ba8" ] || die "solidpc copy sha $got"
say "archived on solidpc: $SRCDIR/qad.gguf sha $got"
claude-hooks-res acquire eleven2go:shared -r "bm-897 add the 26B IQ2_XXS QAD s600 tag (owner 23:00)" --ttl 2h >/dev/null 2>&1 \
  || die "eleven2go claim refused: $(claude-hooks-res status 2>&1 | grep eleven2go | cut -c1-200)"
one 26b mannix/gemma4-26b-agentic:mtp-vision-IQ2_XXS $SRCDIR/qad.gguf g4_26b_agentic_qad_IQ2_XXS gemma-4-26B-A4B-agentic-IQ2_XXS.gguf 128 "$PROJ26" mannix/gemma4-26b-agentic:mtp-vision-Q4_K_M
claude-hooks-res release eleven2go >/dev/null 2>&1
say "KQE_DONE mannix/gemma4-26b-agentic:mtp-vision-IQ2_XXS added"
