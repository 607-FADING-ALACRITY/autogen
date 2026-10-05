#!/usr/bin/env bash
# pod_rebuild.sh: set up ComfyUI on a fresh RunPod pod (runpod/comfyui template, deployed on a
# network volume) from the backup pod_checkpoint.sh made, then download the models the workflows use.
#
# Settings (export them, or put them in front of the command):
#   CIVITAI_TOKEN=xxxxxxxx    required while a Civitai model is missing: civitai.com → Account settings
#                             → API Keys. FinePorn, Stable Yogi and Realistic Snapshot are login-only.
#   BACKUP_TAR=/workspace/backup.tar     default: newest backup*.tar in /workspace (or one folder down)
#   OUTPUTS_TAR=/workspace/outputs.tar   default: newest outputs*.tar there, if any
#   WITH_PORTRAIT=1     also download BiRefNet + Depth Anything 3 (portrait blur in postprocess.json)
#   WITH_KREA2_BASE=1   also download stock Krea 2 Turbo bf16 (26 GB; none of these workflows need it)
#   VERIFY=1            SHA-256 check each model (about a minute per 10 GB on a network volume)
#   RESTART=0           don't restart ComfyUI at the end
#   DRY_RUN=1           show the plan, change nothing
#   COMFY_DIR=/workspace/runpod-slim/ComfyUI
#
# Run in the pod's web terminal once ComfyUI answers on port 8188 (the template's first boot):
#   cd /workspace/runpod-slim/ComfyUI
#   git clone --depth 1 --filter=blob:none --sparse --branch claude/beautiful-goodall-9aeb7g \
#       https://github.com/607-FADING-ALACRITY/autogen.git sm-workflows
#   cd sm-workflows && git sparse-checkout set comfyui-workflows
#   CIVITAI_TOKEN=your_key bash comfyui-workflows/tools/pod_rebuild.sh
#
# Safe to rerun: finished downloads are skipped and interrupted ones resume.

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
COMFY_DIR="${COMFY_DIR:-$WORKSPACE/runpod-slim/ComfyUI}"
VENV_DIR="${VENV_DIR:-$COMFY_DIR/.venv-cu128}"
ARGS_FILE="$WORKSPACE/runpod-slim/comfyui_args.txt"
LOG_FILE="$WORKSPACE/runpod-slim/comfyui.log"
REPO_WF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this repo's comfyui-workflows/
CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"
DRY_RUN="${DRY_RUN:-0}"; VERIFY="${VERIFY:-0}"; RESTART="${RESTART:-1}"
WITH_PORTRAIT="${WITH_PORTRAIT:-0}"; WITH_KREA2_BASE="${WITH_KREA2_BASE:-0}"
TMP_DIR="$(mktemp -d)"; trap 'rm -rf "$TMP_DIR"' EXIT

# path under models/ | source | url | exact bytes | sha256
MODELS=(
  "diffusion_models/fineporn_v5_fp8.safetensors|civitai|https://civitai.com/api/download/models/3361846?fileId=3249668|13141769680|1be696a2e3730de539e17fba5e3f8ce1cdb3d5f5653708d90b2ed0d6679f5879"
  "diffusion_models/realismByStableYogi_v30_fp8_scaled.safetensors|civitai|https://civitai.com/api/download/models/3329215?fileId=3218460|14128290360|3ec3c673d467e0832ad1324fafa8eed5cde9e9f2e250b399d07245e63c904191"
  "loras/RealisticSnapshotKrea2V2.safetensors|civitai|https://civitai.com/api/download/models/3371723?fileId=3260432|1562410304|bc064297da569e9f25fe362d8210b291093fad5fa4f59ea982fa4f7c4151e2bc"
  "text_encoders/qwen3vl_4b_bf16.safetensors|hf|https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors|8875719384|36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34"
  "vae/qwen_image_vae.safetensors|hf|https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors|253806246|a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
  "detection/mediapipe_face_fp32.safetensors|hf|https://huggingface.co/Comfy-Org/mediapipe/resolve/main/detection/mediapipe_face_fp32.safetensors|5423900|a98c4806081d40eba35102a0f6dc0000c2e1388b72cf24e691703d0605bd888a"
)
# Optional: flag that turns it on | the model entry
OPTIONAL=(
  "WITH_PORTRAIT|background_removal/birefnet.safetensors|hf|https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors|444473596|9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"
  "WITH_PORTRAIT|geometry_estimation/depth_anything_3_mono_large.safetensors|hf|https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/depth_anything_3_mono_large.safetensors|1336748056|9b44eda5bedba5b4e125686fdb79d1db309c1b9785277576eb930f885b008f96"
  "WITH_KREA2_BASE|diffusion_models/krea2_turbo_bf16.safetensors|hf|https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_bf16.safetensors|26283332608|78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3"
)
for o in "${OPTIONAL[@]}"; do
  flag=${o%%|*}
  [ "${!flag:-0}" = 1 ] && MODELS+=("${o#*|}")
done
WORKFLOWS=(realism-pass/realism_pass.json snapshot/snapshot.json postprocess/postprocess.json
           fineporn-faceswap/fineporn_faceswap.json fineporn-faceswap/faceswap_only.json)
TEMPLATE_NODES=(ComfyUI-Manager ComfyUI-KJNodes Civicomfy ComfyUI-RunpodDirect)

FAILS=()
WARNS=()
step() { printf '\n[%s] %s\n' "$1" "$2"; }
ok()   { printf '   ok    %s\n' "$1"; }
warn() { printf '   WARN  %s\n' "$1"; WARNS+=("$1"); }
fail() { printf '   FAIL  %s\n' "$1"; FAILS+=("$1"); }
human(){ numfmt --to=si --suffix=B "$1" 2>/dev/null || echo "${1}B"; }
run()  { if [ "$DRY_RUN" = 1 ]; then echo "   (dry run) $*"; else "$@"; fi; }
newest() { find "$WORKSPACE" -maxdepth 2 -type f -name "$1" -not -path '*/_checkpoint/*' -printf '%T@ %p\n' 2>/dev/null \
             | sort -rn | head -1 | cut -d' ' -f2-; }

# A model file is good when it has the exact published size and starts like a safetensors file
# (8-byte header length, then '{'). That catches the 0-byte and error-page files a failed download leaves.
model_ok() {  # file bytes sha
  [ -f "$1" ] || return 1
  [ "$(stat -c %s "$1")" = "$2" ] || return 1
  [ "$(head -c 9 "$1" | tail -c 1)" = "{" ] || return 1
  if [ "$VERIFY" = 1 ]; then [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$3" ] || return 1; fi
  return 0
}

# --------------------------------------------------------------------------------------------
step 1/6 "Checking this pod"
if [ ! -f "$COMFY_DIR/main.py" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "   ComfyUI isn't installed at $COMFY_DIR yet. The template copies it in on first boot:"
  echo "   wait until port 8188 opens, then rerun. (Different folder? Rerun with COMFY_DIR=/path.)"
  exit 1
fi
WS_REAL="$(realpath -m "$WORKSPACE")"
WS_MOUNT=$(findmnt -n -o TARGET --target "$WS_REAL" 2>/dev/null | head -1)
if [ -z "$WS_MOUNT" ]; then
  [ "$(stat -L -c %d "$WS_REAL")" != "$(stat -L -c %d /)" ] && WS_MOUNT="$WS_REAL" || WS_MOUNT="/"
fi
if [ "$WS_MOUNT" = "/" ]; then
  echo "   STOP: $WORKSPACE is not a volume on this pod, so everything this script downloads would be"
  echo "   deleted on Stop, just like last time. Deploy the pod with your network volume attached"
  echo "   (Pods → Deploy → Network Volume → pick it) and run this script there."
  exit 1
fi
ok "$WORKSPACE is a volume ($(findmnt -n -o SOURCE,FSTYPE --target "$WS_REAL" 2>/dev/null | head -1 | xargs)); downloads will survive Stop"
ok "ComfyUI at $COMFY_DIR (version $(sed -n 's/^__version__ = //p' "$COMFY_DIR/comfyui_version.py" 2>/dev/null | tr -d '"'))"
command -v curl >/dev/null || { echo "   curl is missing: apt-get update && apt-get install -y curl"; exit 1; }
[ "$DRY_RUN" = 1 ] && echo "   DRY RUN: nothing will be changed"
cd "$COMFY_DIR" || exit 1

# --------------------------------------------------------------------------------------------
step 2/6 "Restoring your backup"
BACKUP_TAR="${BACKUP_TAR:-$(newest 'backup*.tar')}"
OUTPUTS_TAR="${OUTPUTS_TAR:-$(newest 'outputs*.tar')}"
RESUME_MD=""
if [ -z "$BACKUP_TAR" ] || [ ! -f "$BACKUP_TAR" ]; then
  warn "No backup*.tar found in $WORKSPACE. Upload it there (FileBrowser on port 8080) and rerun; without it, your own LoRAs won't be restored."
elif ! LISTING=$(tar -tf "$BACKUP_TAR" 2>&1); then
  fail "$BACKUP_TAR is damaged or incomplete (did the upload finish?): $(echo "$LISTING" | tail -1)"
else
  ok "found $BACKUP_TAR ($(human "$(stat -c %s "$BACKUP_TAR")"))"
  if echo "$LISTING" | grep -qx 'RESUME.md'; then
    tar -xOf "$BACKUP_TAR" RESUME.md > "$TMP_DIR/RESUME.md" && RESUME_MD="$TMP_DIR/RESUME.md"
  fi
  # --skip-old-files: a rerun never overwrites a LoRA or workflow you changed since the first run
  if run tar -xf "$BACKUP_TAR" -C "$COMFY_DIR" --no-same-owner --skip-old-files --exclude=RESUME.md --exclude=pip-freeze.txt; then
    while IFS= read -r f; do ok "in place: $f"; done < <(echo "$LISTING" | grep -E '^models/.+[^/]$')
    n_wf=$(echo "$LISTING" | grep -cE '^user/default/workflows/.+\.json$')
    [ "$n_wf" -gt 0 ] && ok "restored $n_wf saved workflows"
    # Custom nodes that weren't on git come back from the tar; install their Python deps
    for d in $(echo "$LISTING" | sed -n 's|^custom_nodes/\([^/]*\)/$|\1|p' | sort -u); do
      ok "restored custom node $d"
      if [ -f "custom_nodes/$d/requirements.txt" ]; then
        run "$VENV_DIR/bin/python" -m pip install -q -r "custom_nodes/$d/requirements.txt" \
          || warn "pip install failed for custom node $d; it may not load"
      fi
    done
  else
    fail "Couldn't extract $BACKUP_TAR into $COMFY_DIR"
  fi
fi
if [ -n "$OUTPUTS_TAR" ] && [ -f "$OUTPUTS_TAR" ]; then
  if tar -tf "$OUTPUTS_TAR" >/dev/null 2>&1 && run tar -xf "$OUTPUTS_TAR" -C "$COMFY_DIR" --no-same-owner --skip-old-files; then
    ok "restored outputs from $OUTPUTS_TAR"
  else
    fail "$OUTPUTS_TAR is damaged or incomplete; outputs not restored"
  fi
fi

# --------------------------------------------------------------------------------------------
step 3/6 "Custom nodes and workflows"
if [ -d "$REPO_WF/postprocess/comfyui-postfx" ]; then
  run rm -rf custom_nodes/comfyui-postfx
  run cp -r "$REPO_WF/postprocess/comfyui-postfx" custom_nodes/comfyui-postfx && ok "installed comfyui-postfx (latest from the repo)"
else
  fail "comfyui-postfx not found next to this script ($REPO_WF/postprocess/comfyui-postfx). Run it from the sm-workflows checkout."
fi
for n in "${TEMPLATE_NODES[@]}"; do
  [ -d "custom_nodes/$n" ] && ok "template node present: $n" || warn "template node missing: $n (the template normally installs it; not needed by these workflows)"
done
run mkdir -p user/default/workflows
for w in "${WORKFLOWS[@]}"; do
  name=$(basename "$w")
  if [ -f "user/default/workflows/$name" ]; then ok "workflow kept: $name"
  elif [ -f "$REPO_WF/$w" ]; then run cp "$REPO_WF/$w" "user/default/workflows/$name" && ok "workflow added: $name"
  fi
done

# --------------------------------------------------------------------------------------------
step 4/6 "Models"
TODO=()
for m in "${MODELS[@]}"; do
  IFS='|' read -r path src url bytes sha <<< "$m"
  if model_ok "models/$path" "$bytes" "$sha"; then
    ok "have $path"
  else
    [ -f "models/$path" ] && echo "   bad   $path is $(human "$(stat -c %s "models/$path")"), expected $(human "$bytes"); will re-download"
    TODO+=("$m")
  fi
done

if printf '%s\n' "${TODO[@]}" | grep -q '|civitai|'; then
  if [ -z "$CIVITAI_TOKEN" ]; then
    fail "CIVITAI_TOKEN isn't set, so the Civitai models are skipped. Get a key at civitai.com → Account settings → API Keys, then rerun with CIVITAI_TOKEN=your_key"
  else
    probe=$(printf '%s\n' "${TODO[@]}" | grep '|civitai|' | head -1 | cut -d'|' -f3)
    code=$(curl -sS -o /dev/null -L -r 0-15 -w '%{http_code}' --max-time 60 "$probe&token=$CIVITAI_TOKEN" 2>/dev/null)
    case "$code" in
      200|206) ok "Civitai accepted your token" ;;
      401|403) fail "Civitai rejected CIVITAI_TOKEN (HTTP $code). Check you copied the whole key; Civitai models skipped."; CIVITAI_TOKEN="" ;;
      *)       fail "Couldn't reach Civitai (HTTP ${code:-none}); Civitai models skipped. Rerun later."; CIVITAI_TOKEN="" ;;
    esac
  fi
fi

if [ -z "$CIVITAI_TOKEN" ] && [ ${#TODO[@]} -gt 0 ]; then
  mapfile -t TODO < <(printf '%s\n' "${TODO[@]}" | grep -v '|civitai|')
fi
need=0
for m in "${TODO[@]}"; do
  IFS='|' read -r path src url bytes sha <<< "$m"
  part=0; [ -f "models/$path.part" ] && part=$(stat -c %s "models/$path.part")
  need=$((need + bytes - part))
done
free=$(df -B1 --output=avail "$WS_REAL" | tail -1 | tr -d ' ')
if [ ${#TODO[@]} -gt 0 ]; then
  echo "   to download: $(human "$need"); free on $WORKSPACE: $(human "$free")"
  if [ "$need" -gt $((free - 2000000000)) ]; then
    fail "Not enough space: need $(human "$need") plus 2 GB headroom. Grow the network volume (Storage → your volume → Edit) and rerun."
    TODO=()
  fi
fi

i=0
for m in "${TODO[@]}"; do
  IFS='|' read -r path src url bytes sha <<< "$m"
  i=$((i + 1))
  dest="models/$path"; part="$dest.part"
  fetch_url="$url"
  [ "$src" = civitai ] && fetch_url="$url&token=$CIVITAI_TOKEN"
  echo
  echo "   Downloading $i/${#TODO[@]}: $path ($(human "$bytes"))"
  [ "$DRY_RUN" = 1 ] && { echo "   (dry run) from ${url}"; continue; }
  mkdir -p "$(dirname "$dest")"
  [ -f "$part" ] && [ "$(stat -c %s "$part")" -gt "$bytes" ] && rm -f "$part"
  for attempt in 1 2 3 4 5 6; do
    have=0; [ -f "$part" ] && have=$(stat -c %s "$part")
    [ "$have" = "$bytes" ] && break
    curl -fL --connect-timeout 30 --speed-limit 102400 --speed-time 120 -C - --progress-bar \
      -o "$part" "$fetch_url" && break
    rc=$?
    echo "   attempt $attempt failed (curl exit $rc); retrying in $((2 ** attempt))s..."
    sleep $((2 ** attempt))
  done
  if model_ok "$part" "$bytes" "$sha"; then
    mv -f "$part" "$dest" && ok "$path"
  else
    got=0; [ -f "$part" ] && got=$(stat -c %s "$part")
    fail "$path: download incomplete or wrong file ($(human "$got") of $(human "$bytes")). Rerun to resume."
  fi
done

# Everything the old pod had, according to the RESUME.md inside the backup
if [ -n "$RESUME_MD" ]; then
  while IFS= read -r path; do
    case "$path" in */*) ;; *) continue;; esac
    [ -s "models/$path" ] && continue
    printf '%s\n' "${MODELS[@]}" | grep -q "^$path|" && continue
    flag=$(printf '%s\n' "${OPTIONAL[@]}" | grep "|$path|" | cut -d'|' -f1)
    if [ -n "$flag" ]; then warn "not downloaded: models/$path (your old pod had it; rerun with $flag=1 to get it)"
    else warn "not restored: models/$path (it wasn't in the backup and this script has no download source for it)"; fi
  done < <(sed -n 's/^| [^|]* | \(.*\) |$/\1/p' "$RESUME_MD")
fi

# --------------------------------------------------------------------------------------------
step 5/6 "Restarting ComfyUI so it loads the nodes"
PORT=8188
if [ "$RESTART" != 1 ] || [ "$DRY_RUN" = 1 ]; then
  echo "   skipped (RESTART=0 or dry run). Restart from ComfyUI-Manager when you're ready."
else
  PIDS=(); ARGV=(); CWD="$COMFY_DIR"
  for p in /proc/[0-9]*; do
    mapfile -d '' -t argv 2>/dev/null < "$p/cmdline" || continue
    [ ${#argv[@]} -gt 1 ] || continue
    case "$(basename "${argv[0]}")" in python*) ;; *) continue;; esac
    is_main=no
    for a in "${argv[@]:1}"; do case "$a" in main.py|*/main.py) is_main=yes;; esac; done
    [ "$is_main" = yes ] || continue
    c=$(readlink "$p/cwd" 2>/dev/null || true)
    [ "$(realpath -m "$c")" = "$(realpath -m "$COMFY_DIR")" ] || continue
    PIDS+=("${p#/proc/}"); ARGV=("${argv[@]}"); CWD="$c"
  done
  if [ ${#ARGV[@]} -eq 0 ]; then
    # Same arguments the template's start.sh uses
    ARGV=("$VENV_DIR/bin/python" main.py --listen 0.0.0.0 --port 8188 --enable-cors-header)
    [ -s "$ARGS_FILE" ] && while read -r a; do [ -n "$a" ] && ARGV+=("$a"); done < <(grep -v '^#' "$ARGS_FILE" | tr ' ' '\n')
  fi
  # The template starts ComfyUI with a bare "python" from its activated venv; use the venv's path
  case "${ARGV[0]}" in */*) ;; *) ARGV[0]="$VENV_DIR/bin/python" ;; esac
  for ((k = 1; k < ${#ARGV[@]} - 1; k++)); do [ "${ARGV[$k]}" = "--port" ] && PORT="${ARGV[$((k + 1))]}"; done
  if [ ${#PIDS[@]} -gt 0 ]; then
    echo "   stopping ComfyUI (pid ${PIDS[*]}); the pod and its files stay up"
    kill -TERM "${PIDS[@]}" 2>/dev/null
    for _ in $(seq 1 30); do alive=0; for pid in "${PIDS[@]}"; do [ -d "/proc/$pid" ] && alive=1; done; [ $alive = 0 ] && break; sleep 1; done
    for pid in "${PIDS[@]}"; do [ -d "/proc/$pid" ] && kill -KILL "$pid" 2>/dev/null; done
  fi
  echo "   starting: ${ARGV[*]}  (log: $LOG_FILE)"
  # Own session + nohup: closing the web terminal won't take ComfyUI down with it
  (cd "$CWD" && export VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$PATH" && exec setsid nohup "${ARGV[@]}") \
    >> "$LOG_FILE" 2>&1 < /dev/null &
  up=no
  for _ in $(seq 1 150); do
    sleep 2
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/system_stats" && { up=yes; break; }
  done
  if [ "$up" = yes ]; then
    ok "ComfyUI is back on port $PORT"
    if curl -sf "http://127.0.0.1:$PORT/object_info/PostFXSaveJPEG" | grep -q PostFXSaveJPEG; then
      ok "comfyui-postfx nodes loaded"
    else
      fail "ComfyUI is up but comfyui-postfx didn't load. Look for its traceback: grep -n -A20 postfx $LOG_FILE"
    fi
  else
    fail "ComfyUI didn't come back within 5 minutes. Last lines of $LOG_FILE:"
    tail -20 "$LOG_FILE" 2>/dev/null | sed 's/^/          /'
  fi
fi

# --------------------------------------------------------------------------------------------
step 6/6 "Summary"
for m in "${MODELS[@]}"; do
  IFS='|' read -r path src url bytes sha <<< "$m"
  if model_ok "models/$path" "$bytes" "$sha"; then printf '   %-8s %s\n' "ready" "$path"
  else printf '   %-8s %s\n' "MISSING" "$path"; fi
done
if [ ${#WARNS[@]} -gt 0 ]; then
  echo
  echo "   Warnings:"
  printf '    - %s\n' "${WARNS[@]}"
fi
echo
if [ ${#FAILS[@]} -gt 0 ]; then
  echo "   >>> NOT DONE. Fix these and rerun (finished downloads are kept):"
  printf '    - %s\n' "${FAILS[@]}"
  exit 1
fi
echo "   >>> DONE. Open ComfyUI on port $PORT; your workflows are in the Workflows sidebar."
echo "   Your files live on the network volume: Stop or Terminate this pod when you're finished, and"
echo "   tomorrow deploy a new pod on the same volume. Nothing needs reinstalling."
