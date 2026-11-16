#!/bin/bash
# kfac_ollama_bs2.sh -- RUNS ON bs2. Pushes every HF-published K-FAC / seeded-QAD tier of both ladders to ollama.com as
# the four variants the 26B already has (text, vision-, mtp-, mtp-vision-) + :latest = Q4_K_M, via the bs2 xollama
# (:22434, store /mnt/sdc/ollama/models, key = mannix). Owner 2026-11-16 22:28 CET: "push the kfac tiers to ollama too
# ... republish everything on ollama for both 20b and 26b as they come".
#   20b: ManniX-ITA/gemma-4-20B-v9-agentic-GGUF  -> mannix/gemma-4-20B-v9-agentic   (PRIVATE repo on ollama.com)
#   26b: ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF -> mannix/gemma4-26b-agentic
# Recipe = the public 26B tags read back from registry.ollama.ai (2026-11-16): config renderer gemma4 / parser gemma4;
# params {min_p 0, presence_penalty 0.025, repeat_penalty 1, temperature 0.8, top_k 64, top_p 0.95}; mtp- variants add
# DRAFT gemma-4-26B-A4B-it-assistant Q8_0, draft_num_predict 2, think_budget high + the stop phrase; vision- variants
# add the model's own mmproj (downloaded once from its HF repo). Preflight (feedback_ollama_publish_preflight): RENDERER
# and PARSER declared -> no jinja-mined stop; host xollama 0.40.13 knows gemma4; the store is the service's XOLLAMA_MODELS.
# Per tier (every 5 min, 48 h): needs <ladder>/<TIER>/.published_hf (hub sha verified) and no .published_ollama. The
# source file is the local kfac.gguf / qad.gguf; when the publisher already removed it, the tier is downloaded from HF
# (curl -4, resumable, Bearer token from the HF cache) to the root-fs relief dir and sha-checked against .published_hf,
# then deleted after the push. create -> push (positive markers: exit 0 + "success", never "signed in") -> read back the
# registry manifest (public repo: anonymous; private: skipped, push output is the evidence) -> rm the local tags ->
# sweep orphan blobs (XOLLAMA_NOPRUNE=1 on the service; refuses when the reference set is empty). Writes
# .published_ollama (tags) so the HF publisher / archiver may reclaim the bs2 copy. Never touches a run artifact.
# Sentinels KO_START / KO_TIER / KO_TAG / KO_FAIL / KO_SKIP / KO_DONE in runs/kfac_ollama_bs2.log.
set -uo pipefail
RUNS=/mnt/sdc/omk-model-optimizer/runs
LOG=$RUNS/kfac_ollama_bs2.log
export XOLLAMA_HOST=127.0.0.1:22434
XO=/usr/local/bin/xollama
STORE=/mnt/sdc/ollama/models
DRAFT=/mnt/sdc/ml/gguf/gemma-4-26B-A4B-it-assistant.Q8_0.gguf
DL=/srv/ml/tmp_sdc_relief/ollama_dl; mkdir -p $DL
AUX=$RUNS/ollama_aux; mkdir -p $AUX
HOURS=${HOURS:-48}
TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null); [ -n "$TOKEN" ] || { echo "no HF token"; exit 1; }
# the token never goes on a command line (ps shows argv to every user): curl reads it from a 0600 config file
CURLRC=$AUX/.curlrc; ( umask 077; printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" > $CURLRC ); unset TOKEN
say() { echo "[ko $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
MSG='
I have used my thinking budget. I must stop analysing now and act on what I have: make the tool call, or give a short final answer if no call is needed.
I'"'"'ll be more terse and concise now and maybe I need to consider a different approach.
'
# which -> hf repo, prefix, ollama target, ladder dir, mmproj file name on the hub
cfg() {
  case $1 in
    20b) HF=ManniX-ITA/gemma-4-20B-v9-agentic-GGUF; PFX=gemma-4-20B-v9-agentic; OL=mannix/gemma-4-20B-v9-agentic
         LAD=$RUNS/g4_20b_kfac/ladder; MM=mmproj-gemma-4-20B-v9-agentic-F16.gguf; PUBLIC=0 ;;
    26b) HF=ManniX-ITA/gemma-4-26B-A4B-agentic-GGUF; PFX=gemma-4-26B-A4B-agentic; OL=mannix/gemma4-26b-agentic
         LAD=$RUNS/g4_agentic_kfac/ladder; MM=mmproj-gemma-4-26B-A4B-agentic-BF16.gguf; PUBLIC=1 ;;
  esac
}
hub_get() {  # <repo> <file> <out>  resumable single-stream download with the token (private repos too)
  curl -4 -sS -L -C - --retry 5 --retry-delay 20 -K $CURLRC "https://huggingface.co/$1/resolve/main/$2" -o "$3"
}
hub_sha() {  # <repo> <file> -> lfs sha256 via the API (paths-info names it lfs.oid)
  curl -4 -sS -K $CURLRC "https://huggingface.co/api/models/$1/paths-info/main" \
    -X POST -H "Content-Type: application/json" -d "{\"paths\":[\"$2\"],\"expand\":true}" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); l=d[0].get("lfs") or {}; print(l.get("oid") or l.get("sha256") or "")' 2>/dev/null
}
ensure_aux() {  # the mmproj of this model, once, sha-verified
  local out=$AUX/$MM
  [ -s $out ] && [ -s $out.ok ] && return 0
  local want; want=$(hub_sha $HF $MM); [ ${#want} = 64 ] || { say "KO_FAIL $W no hub sha for $MM"; return 1; }
  say "$W: downloading $MM"
  hub_get $HF $MM $out || { say "KO_FAIL $W mmproj download"; return 1; }
  local got; got=$(sha256sum $out | cut -d' ' -f1)
  [ "$got" = "$want" ] || { say "KO_FAIL $W mmproj sha $got != hub $want"; rm -f $out; return 1; }
  echo "$got" > $out.ok; say "$W: $MM ok ($got)"
}
modelfile() {  # <variant> <gguf> -> stdout
  local v=$1 g=$2
  echo "FROM $g"
  case $v in *mtp*) echo "DRAFT $DRAFT";; esac
  case $v in *vision*) echo "FROM $AUX/$MM";; esac
  printf 'RENDERER gemma4\nPARSER gemma4\nPARAMETER temperature 0.8\nPARAMETER top_k 64\nPARAMETER top_p 0.95\nPARAMETER min_p 0\nPARAMETER presence_penalty 0.025\nPARAMETER repeat_penalty 1\n'
  case $v in *mtp*) printf 'PARAMETER draft_num_predict 2\nPARAMETER think_budget high\nPARAMETER think_budget_message "%s"\n' "$MSG";; esac
}
push_tag() {  # <tag> <modelfile> -> 0 on verified push
  local tag=$1 mf=$2 out=$AUX/push.$(echo $tag | tr '/:' '__').log
  $XO create "$tag" -f "$mf" > $out.create 2>&1 || { say "KO_FAIL create $tag: $(tail -3 $out.create | tr '\n' ' ')"; return 1; }
  $XO push "$tag" > $out 2>&1; local rc=$?
  # positive markers only (ollama_push_gemma4.sh): "success" + the "You can find your model at" URL; the
  # "You need to be signed in" warning appears on successful pushes too and is NOT a failure signal
  local ok=0; grep -qE "^success" $out && grep -q "You can find your model at" $out && ok=1
  if [ $rc != 0 ] || [ $ok != 1 ]; then
    say "KO_FAIL push $tag rc=$rc markers=$ok: $(tail -3 $out | tr -d '\r' | tr '\n' ' ' | cut -c1-240)"; return 1
  fi
  return 0
}
registry_check() {  # <target> <tagname> (public repos only): config has renderer+parser, layers listed
  local A='application/vnd.docker.distribution.manifest.v2+json'
  local m; m=$(curl -4 -s -m 30 -H "Accept: $A" "https://registry.ollama.ai/v2/$1/manifests/$2")
  local c; c=$(echo "$m" | python3 -c 'import json,sys; print(json.load(sys.stdin)["config"]["digest"])' 2>/dev/null)
  [ -n "$c" ] || { echo "no manifest"; return 1; }
  local cfgj; cfgj=$(curl -4 -s -m 30 -L "https://registry.ollama.ai/v2/$1/blobs/$c")
  if ! { echo "$cfgj" | grep -q '"renderer":"gemma4"' && echo "$cfgj" | grep -q '"parser":"gemma4"'; }; then echo "config without renderer/parser: $cfgj"; return 1; fi
  echo "$m" | python3 -c 'import json,sys; m=json.load(sys.stdin); print(" ".join(l["mediaType"].split(".")[-1]+":"+l["digest"][7:15] for l in m["layers"]))'
}
sweep_blobs() {  # orphan blobs of the store; refuses on an empty reference set
  local refs; refs=$(grep -rhoE 'sha256:[0-9a-f]{64}' $STORE/manifests 2>/dev/null | sort -u | sed 's/sha256:/sha256-/')
  [ -n "$refs" ] || { say "sweep: reference set EMPTY, refusing (manifests dir $STORE/manifests)"; return 0; }
  local n=0 b=0
  for f in $STORE/blobs/sha256-*; do
    [ -e "$f" ] || continue
    echo "$refs" | grep -qx "$(basename $f)" && continue
    b=$((b + $(stat -c %s "$f"))); rm -f "$f"; n=$((n + 1))
  done
  say "sweep: $n orphan blobs removed ($((b / 1073741824)) GiB)"
}
one_tier() {  # $W set; <tierdir>
  local D=$1 T=$(basename $1)
  local parts; read -r -a parts < $D/.published_hf; local sha=${parts[0]} method=${parts[2]:-kfac}
  local name=$PFX-$T.gguf
  local src=$D/$method.gguf tmp=""
  if [ ! -s "$src" ]; then
    tmp=$DL/$name; say "$W $T: local copy gone, downloading $name from the hub ($(df --output=avail -BG /srv/ml | tail -1 | tr -dc 0-9)G free on root fs)"
    hub_get $HF $name $tmp || { say "KO_FAIL $W $T download"; return 1; }
    local got; got=$(sha256sum $tmp | cut -d' ' -f1)
    [ "$got" = "$sha" ] || { say "KO_FAIL $W $T downloaded sha $got != published $sha"; rm -f $tmp; return 1; }
    src=$tmp
  else
    local got; got=$(cut -d' ' -f1 $D/$method.gguf.sha256 2>/dev/null); [ "$got" = "$sha" ] || { say "KO_FAIL $W $T local sidecar $got != published $sha"; return 1; }
  fi
  ensure_aux || return 1
  local free; free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
  [ "$free" -ge 40 ] || { say "KO_SKIP $W $T: /mnt/sdc ${free}G < 40G for the store copy"; return 2; }
  say "KO_TIER $W $T ($method) from $src -> $OL"
  local tags="" fail=0
  for v in "" vision- mtp- mtp-vision-; do
    local tag=$OL:$v$T mf=$AUX/Modelfile.$W.$v$T
    modelfile "$v" "$src" > $mf
    if push_tag $tag $mf; then
      local rb=""; if [ $PUBLIC = 1 ]; then rb=$(registry_check $OL $v$T) || { say "KO_FAIL $tag registry read-back: $rb"; fail=1; break; }; fi
      say "KO_TAG $tag pushed${rb:+ [$rb]}"; tags="$tags $tag"
    else fail=1; break; fi
  done
  if [ $fail = 0 ] && [ $T = Q4_K_M ]; then
    if push_tag $OL:latest $AUX/Modelfile.$W.Q4_K_M; then say "KO_TAG $OL:latest pushed (= Q4_K_M)"; tags="$tags $OL:latest"; else fail=1; fi
  fi
  for t in $tags; do $XO rm "$t" > /dev/null 2>&1; done
  sweep_blobs
  [ -n "$tmp" ] && rm -f $tmp && say "$W $T: downloaded copy removed"
  [ $fail = 0 ] || return 1
  echo "$(date -u +%FT%TZ) $sha$tags" > $D/.published_ollama
  say "KO_TIER_DONE $W $T:$tags"
}
say "KO_START $HOURS h; xollama $($XO --version 2>&1 | tail -1); store $STORE; draft $DRAFT"
[ -s $DRAFT ] || { say "KO_FAIL no drafter $DRAFT"; exit 1; }
end=$(( $(date +%s) + HOURS * 3600 ))
while [ $(date +%s) -lt $end ]; do
  for W in 20b 26b; do
    cfg $W
    for D in $LAD/*/; do
      D=${D%/}; [ -e $D/.published_hf ] || continue; [ -e $D/.published_ollama ] && continue
      one_tier $D
    done
  done
  sleep 300
done
say "KO_DONE $HOURS h elapsed"
