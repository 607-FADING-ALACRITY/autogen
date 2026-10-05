#!/usr/bin/env bash
# pod_checkpoint.sh: run on the RunPod pod BEFORE you press Stop.
#
# Read-only, except that it writes into /workspace/_checkpoint/. It never stops, restarts or
# kills anything, and it copies no tokens or settings files.
#
#   1. Checks that /workspace is a separate volume. On RunPod, Stop keeps the volume and resets
#      everything else (the container disk) to the template image. Terminate deletes both.
#   2. Finds anything created outside /workspace since the pod started (Stop deletes it) and
#      anything ComfyUI loads from outside /workspace.
#   3. Writes /workspace/_checkpoint/RESUME.md: the exact launch command, ComfyUI version,
#      custom nodes with their git commits, every model file with its size, saved workflows.
#   4. Packs what you can't re-download (your own LoRAs, saved workflows, custom nodes that
#      aren't git repos) into /workspace/_checkpoint/backup_<date>.tar, to copy off the pod.
#      On a pod with no volume it also prints browser download links for the tar, served by
#      ComfyUI's own /view endpoint through the pod's RunPod proxy address.
#
# Run in the pod's web terminal:
#   bash /workspace/runpod-slim/ComfyUI/sm-workflows/comfyui-workflows/tools/pod_checkpoint.sh
#
# Optional overrides (no API keys needed):
#   COMFY_DIR=/workspace/runpod-slim/ComfyUI   your ComfyUI folder
#   OWN_LORAS='dcn_*'                          name pattern of LoRAs you trained yourself
#   WORKSPACE=/workspace                       the persistent volume's mount point
#   INCLUDE_OUTPUTS=1                          also pack ComfyUI's output folder (your generations)

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
COMFY_DIR="${COMFY_DIR:-$WORKSPACE/runpod-slim/ComfyUI}"
OWN_LORAS="${OWN_LORAS:-dcn_*}"
INCLUDE_OUTPUTS="${INCLUDE_OUTPUTS:-0}"
OUT_DIR="$WORKSPACE/_checkpoint"
STAMP="$(date +%Y%m%d_%H%M)"
RESUME="$OUT_DIR/RESUME.md"
MODEL_EXT_RE='.*\.(safetensors|ckpt|pt|pth|gguf|bin|onnx|sft)$'
BIG=$((10 * 1024 * 1024))   # new files outside the volume above this size block the Stop

BLOCKERS=()   # reasons it is not safe to stop yet
WARNINGS=()   # things that will be slower or need a manual step tomorrow

step() { printf '\n[%s] %s\n' "$1" "$2"; }
ok()   { printf '   ok    %s\n' "$1"; }
warn() { printf '   WARN  %s\n' "$1"; WARNINGS+=("$1"); }
block(){ printf '   STOP  %s\n' "$1"; BLOCKERS+=("$1"); }
human(){ numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "${1}B"; }
under_workspace() { case "$(realpath -m "$1")" in "$WS_REAL"/*|"$WS_REAL") return 0;; *) return 1;; esac; }

if [ ! -d "$COMFY_DIR" ]; then
  echo "ComfyUI folder not found: $COMFY_DIR"
  echo "Rerun with: COMFY_DIR=/path/to/ComfyUI bash $0"
  exit 1
fi
WS_REAL="$(realpath -m "$WORKSPACE")"
if ! mkdir -p "$OUT_DIR"; then
  echo "Can't create $OUT_DIR (disk full or read-only?). Check: df -h $WORKSPACE"
  exit 1
fi

# --------------------------------------------------------------------------------------------
step 1/5 "Does $WORKSPACE survive Stop?"
# Resolve symlinks first: on some templates /workspace is a link to the real volume mount.
WS_MOUNT=$(findmnt -n -o TARGET --target "$WS_REAL" 2>/dev/null | head -1)
WS_FS=$(findmnt -n -o SOURCE,FSTYPE,SIZE --target "$WS_REAL" 2>/dev/null | head -1 | xargs)
[ "$WS_REAL" != "$WORKSPACE" ] && echo "   $WORKSPACE is a link to $WS_REAL"
echo "   $WS_REAL sits on the filesystem mounted at ${WS_MOUNT:-?} (${WS_FS:-unknown})"
if [ -n "$WS_MOUNT" ]; then
  [ "$WS_MOUNT" != "/" ] && PERSISTENT=yes || PERSISTENT=no
else
  [ "$(stat -L -c %d "$WS_REAL")" != "$(stat -L -c %d /)" ] && PERSISTENT=yes || PERSISTENT=no
fi
if [ "$PERSISTENT" = yes ]; then
  ok "$WORKSPACE is its own volume. Stop keeps it."
else
  printf '   STOP  %s\n' "$WORKSPACE is on the container disk, not a volume: this pod has no volume disk, so Stop deletes EVERYTHING on it."
fi
if under_workspace "$COMFY_DIR"; then ok "ComfyUI is inside $WORKSPACE: $COMFY_DIR"
else block "ComfyUI lives outside $WORKSPACE ($COMFY_DIR), on the disk Stop wipes."; fi

# --------------------------------------------------------------------------------------------
step 2/5 "Looking for anything that lives outside $WORKSPACE"

# The running ComfyUI: launch command, working dir, Python environment (no pgrep: slim images lack it)
LAUNCH_LINES=()
VENV_DIR=""
PORT=8188
OUTPUT_DIR="$COMFY_DIR/output"
for p in /proc/[0-9]*; do
  # Only the Python process itself (argv[0] is python, one argument is main.py), not shells wrapping it
  mapfile -d '' -t argv 2>/dev/null < "$p/cmdline" || continue
  [ ${#argv[@]} -gt 1 ] || continue
  case "$(basename "${argv[0]}")" in python*) ;; *) continue;; esac
  is_main=no
  for a in "${argv[@]:1}"; do case "$a" in main.py|*/main.py) is_main=yes;; esac; done
  [ "$is_main" = yes ] || continue
  cmd="${argv[*]}"
  cwd=$(readlink "$p/cwd" 2>/dev/null || echo "?")
  case "$cmd $cwd" in *ComfyUI*|*comfy*) ;; *) continue;; esac
  venv=$(tr '\0' '\n' < "$p/environ" 2>/dev/null | sed -n 's/^VIRTUAL_ENV=//p')
  py="${argv[0]}"
  LAUNCH_LINES+=("cd $cwd && $cmd")
  for ((i = 1; i < ${#argv[@]} - 1; i++)); do
    case "${argv[$i]}" in
      --port) PORT="${argv[$((i + 1))]}" ;;
      --output-directory) OUTPUT_DIR="$(cd "$cwd" 2>/dev/null && realpath -m "${argv[$((i + 1))]}")" ;;
      --base-directory) OUTPUT_DIR="$(cd "$cwd" 2>/dev/null && realpath -m "${argv[$((i + 1))]}")/output" ;;
    esac
  done
  [ -n "$venv" ] && VENV_DIR="$venv"
  if [ -z "$VENV_DIR" ]; then case "$py" in */bin/python*) VENV_DIR="$(dirname "$(dirname "$py")")";; esac; fi
done
if [ ${#LAUNCH_LINES[@]} -eq 0 ]; then
  warn "ComfyUI isn't running, so its launch command can't be recorded. The template's start script normally relaunches it."
else
  ok "ComfyUI is running; launch command saved to RESUME.md"
fi
if [ -z "$VENV_DIR" ]; then
  VENV_DIR=$(ls -d "$COMFY_DIR"/.venv* "$COMFY_DIR"/../.venv* 2>/dev/null | head -1 || true)
fi
if [ -n "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
  if under_workspace "$VENV_DIR"; then ok "Python environment is inside $WORKSPACE: $VENV_DIR"
  else warn "Python environment is outside $WORKSPACE ($VENV_DIR). Packages you pip-installed by hand will be gone; pip-freeze.txt lets you reinstall them."; fi
else
  warn "Couldn't find ComfyUI's Python environment; pip-freeze.txt will be skipped."
  VENV_DIR=""
fi

# Model folders redirected elsewhere
if [ -f "$COMFY_DIR/extra_model_paths.yaml" ]; then
  while IFS= read -r path; do
    [ -z "$path" ] && continue
    if under_workspace "$path"; then ok "extra_model_paths.yaml points inside $WORKSPACE: $path"
    else block "extra_model_paths.yaml loads models from outside $WORKSPACE: $path"; fi
  done < <(sed -n 's/^[[:space:]]*base_path:[[:space:]]*//p' "$COMFY_DIR/extra_model_paths.yaml" | tr -d "\"'")
fi

# Symlinks in models/ or custom_nodes/ that point off the volume
while IFS= read -r link; do
  target=$(realpath -m "$link")
  under_workspace "$target" || block "Symlink points off the volume: ${link#"$COMFY_DIR"/} -> $target"
done < <(find "$COMFY_DIR/models" "$COMFY_DIR/custom_nodes" -maxdepth 3 -type l 2>/dev/null)

# Files created on the container disk since the pod started. Those are the ones Stop deletes;
# files that came with the template image come back on Start. ctime can't be faked by wget or cp.
START=""
if [ -r /proc/1/stat ] && [ -r /proc/stat ]; then
  btime=$(awk '/^btime/ {print $2}' /proc/stat)
  ticks=$(sed 's/.*) //' /proc/1/stat | awk '{print $20}')
  hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
  [ -n "$btime" ] && [ -n "$ticks" ] && START=$((btime + ticks / hz))
fi
if [ -z "$START" ]; then
  warn "Couldn't read the pod's start time; skipped the scan for files outside $WORKSPACE."
else
  echo "   pod started $(date -d "@$START" '+%Y-%m-%d %H:%M'); scanning the container disk for files made since (a minute at most)..."
  NEW_DIRS=$(find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /run -o -path /var/log \
      -o -path /var/cache -o -path /var/lib -o -path /root/.cache -o -path "$WS_REAL" \) -prune \
      -o -type f -newerct "@$START" -printf '%s\t%h\n' 2>/dev/null \
    | awk -F'\t' '{d=$2; if (d ~ /^\/usr(\/|$)/) d="/usr"; else if (d ~ /^\/opt(\/|$)/) d="/opt"; s[d]+=$1; n[d]++}
                  END {for (d in s) printf "%d\t%d\t%s\n", s[d], n[d], d}' | sort -rn)
  small=()
  while IFS=$'\t' read -r sz n d; do
    [ -z "$d" ] && continue
    case "$d" in
      /usr|/opt) warn "$(human "$sz") of packages installed under $d since the pod started. Stop removes them; if you installed these by hand, reinstall tomorrow." ;;
      /tmp|/tmp/*) [ "$sz" -ge "$BIG" ] && warn "$d has $(human "$sz") of new files ($n). Stop deletes them; move them into $WORKSPACE if you need them." ;;
      *) if [ "$sz" -ge "$BIG" ]; then block "$d has $(human "$sz") of new files ($n). Stop deletes them: move them into $WORKSPACE."
         else small+=("$d"); fi ;;
    esac
  done <<< "$NEW_DIRS"
  if [ ${#small[@]} -gt 0 ]; then
    warn "Small files changed outside $WORKSPACE since start (Stop resets them): ${small[*]:0:6}$([ ${#small[@]} -gt 6 ] && echo " and $((${#small[@]} - 6)) more")"
  fi
  [ -z "$NEW_DIRS" ] && ok "Nothing created outside $WORKSPACE since the pod started"
fi

# Download caches: harmless to lose, they refill on first use
for c in /root/.cache/huggingface /root/.cache/torch; do
  [ -d "$c" ] || continue
  sz=$(du -sb "$c" 2>/dev/null | cut -f1)
  [ "${sz:-0}" -gt $((100 * 1024 * 1024)) ] && warn "$c holds $(human "$sz"). Stop clears it; anything in it re-downloads on first use tomorrow."
done

# --------------------------------------------------------------------------------------------
step 3/5 "Writing $RESUME"
{
  echo "# Pod checkpoint $(date '+%Y-%m-%d %H:%M %Z')"
  echo
  echo "- Pod: ${RUNPOD_POD_ID:-unknown}. GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
  echo "- $WORKSPACE survives Stop: **$PERSISTENT**"
  echo "- Disk: $(df -h "$WORKSPACE" | awk 'NR==2 {print $3 " used of " $2 ", " $4 " free"}')"
  echo "- ComfyUI: $COMFY_DIR, $(git -C "$COMFY_DIR" describe --tags --always 2>/dev/null || echo 'not a git checkout') (version $(sed -n 's/^__version__ = //p' "$COMFY_DIR/comfyui_version.py" 2>/dev/null | tr -d '"'))"
  echo "- Python environment: ${VENV_DIR:-not found}"
  echo
  echo "## Tomorrow"
  echo
  if [ "$PERSISTENT" = yes ]; then
    echo '1. RunPod console → this pod → **Start**. Never **Terminate**: that deletes the volume.'
    echo "2. The template start script relaunches ComfyUI from the volume. Give it a few minutes, then open port $PORT."
    echo '3. If ComfyUI does not come up, paste the launch command below into the web terminal.'
    echo '4. If RunPod says the machine has no free GPU, start it with 0 GPUs to reach your files, or set up a new pod from the backup tar.'
  else
    echo 'This pod had no volume disk, so Stop deleted everything on it. Rebuild:'
    echo
    echo '1. Create a network volume (Storage → Network Volumes) and deploy the same template on it, so this never happens again.'
    echo '2. Restore your files from the backup tar (your LoRAs, saved workflows, custom nodes that are not on git).'
    echo '3. Re-clone the custom nodes and re-download the models listed below.'
  fi
  echo
  echo "## Launch command (as it was running)"
  echo
  if [ ${#LAUNCH_LINES[@]} -gt 0 ]; then
    for l in "${LAUNCH_LINES[@]}"; do printf '```bash\n%s\n```\n' "$l"; done
  else
    echo 'ComfyUI was not running at checkpoint time.'
  fi
  echo
  echo "Container entrypoint: \`$(tr '\0' ' ' < /proc/1/cmdline 2>/dev/null)\`"
  echo
  echo "## Custom nodes"
  echo
  for d in "$COMFY_DIR"/custom_nodes/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    if git -C "$d" rev-parse --git-dir >/dev/null 2>&1; then
      url=$(git -C "$d" config --get remote.origin.url 2>/dev/null || echo "no remote")
      rev=$(git -C "$d" rev-parse --short HEAD 2>/dev/null || echo "?")
      dirty=$(git -C "$d" status --porcelain 2>/dev/null | head -1)
      echo "- $n: $url @ $rev${dirty:+ (local edits)}"
    else
      echo "- $n: not a git repo, copied into the backup tar"
    fi
  done
  echo
  echo "## Models"
  echo
  echo '| Size | File |'
  echo '|---:|---|'
  find -L "$COMFY_DIR/models" -type f -regextype posix-extended -iregex "$MODEL_EXT_RE" -printf '%s\t%P\n' 2>/dev/null \
    | sort -t$'\t' -k2 | while IFS=$'\t' read -r sz path; do echo "| $(human "$sz") | $path |"; done
  echo
  echo "## Saved workflows"
  echo
  find "$COMFY_DIR/user/default/workflows" -type f -name '*.json' -printf '- %P\n' 2>/dev/null | sort
  echo
  echo "## Outputs"
  echo
  echo "$OUTPUT_DIR: $(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l) files, $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1). Not in the backup tar unless you ran with INCLUDE_OUTPUTS=1."
} > "$RESUME"
ok "RESUME.md written ($(grep -c '^| [0-9]' "$RESUME") model files listed)"

EXTRA=("RESUME.md")
if [ -n "$VENV_DIR" ]; then
  if "$VENV_DIR/bin/python" -m pip freeze > "$OUT_DIR/pip-freeze.txt" 2>/dev/null \
     || { command -v uv >/dev/null && uv pip freeze --python "$VENV_DIR/bin/python" > "$OUT_DIR/pip-freeze.txt" 2>/dev/null; }; then
    ok "pip-freeze.txt written ($(wc -l < "$OUT_DIR/pip-freeze.txt") packages)"
    EXTRA+=("pip-freeze.txt")
  else
    warn "pip freeze failed for $VENV_DIR; package list not saved."
    rm -f "$OUT_DIR/pip-freeze.txt"
  fi
fi

# --------------------------------------------------------------------------------------------
step 4/5 "Packing what can't be re-downloaded"
cd "$COMFY_DIR" || exit 1
PACK=()
mapfile -t OWN < <(find -L models/loras -type f -iname "$OWN_LORAS" 2>/dev/null | sort)
if [ ${#OWN[@]} -eq 0 ]; then
  warn "No LoRAs match '$OWN_LORAS'. If you trained one under another name, rerun with OWN_LORAS='thatname*'."
else
  for f in "${OWN[@]}"; do PACK+=("$f"); ok "your LoRA: $f ($(human "$(stat -L -c %s "$f")"))"; done
fi
if [ -d user/default/workflows ]; then PACK+=("user/default/workflows"); ok "saved workflows: user/default/workflows"; fi
for d in custom_nodes/*/; do
  d=${d%/}
  [ -d "$d" ] || continue
  git -C "$d" rev-parse --git-dir >/dev/null 2>&1 && continue
  sz=$(du -sb "$d" 2>/dev/null | cut -f1)
  if [ "${sz:-0}" -lt $((50 * 1024 * 1024)) ]; then PACK+=("$d"); ok "custom node not on git: $d"
  else warn "custom node $d isn't a git repo and is $(human "$sz"); left out of the tar."; fi
done

TAR=""
if [ ${#PACK[@]} -eq 0 ]; then
  warn "Nothing to pack."
else
  need=$(du -sbL "${PACK[@]}" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  free=$(df -B1 --output=avail "$WORKSPACE" | tail -1 | tr -d ' ')
  if [ "$need" -gt "$free" ]; then
    block "Backup needs $(human "$need") but $WORKSPACE only has $(human "$free") free. Delete old outputs or an old backup_*.tar and rerun."
  elif tar -chf "$OUT_DIR/backup_$STAMP.tar" "${PACK[@]}" -C "$OUT_DIR" "${EXTRA[@]}" 2>"$OUT_DIR/tar_errors.txt"; then
    TAR="$OUT_DIR/backup_$STAMP.tar"
    ok "wrote $TAR ($(human "$(stat -c %s "$TAR")"))"
    rm -f "$OUT_DIR/tar_errors.txt"
  else
    block "tar failed: $(head -3 "$OUT_DIR/tar_errors.txt" | tr '\n' ' ')"
    rm -f "$OUT_DIR/backup_$STAMP.tar"
  fi
fi

OUT_TAR=""
if [ "$INCLUDE_OUTPUTS" = 1 ] && [ -d "$OUTPUT_DIR" ]; then
  need=$(du -sb --exclude=_backup "$OUTPUT_DIR" 2>/dev/null | cut -f1)
  free=$(df -B1 --output=avail "$WORKSPACE" | tail -1 | tr -d ' ')
  if [ "${need:-0}" -gt "$free" ]; then
    block "Outputs need $(human "$need") but only $(human "$free") is free; outputs not packed."
  elif tar -cf "$OUT_DIR/outputs_$STAMP.tar" --exclude=_backup -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")" 2>"$OUT_DIR/tar_errors.txt"; then
    OUT_TAR="$OUT_DIR/outputs_$STAMP.tar"
    ok "wrote $OUT_TAR ($(human "$(stat -c %s "$OUT_TAR")"))"
    rm -f "$OUT_DIR/tar_errors.txt"
  else
    block "outputs tar failed: $(head -3 "$OUT_DIR/tar_errors.txt" | tr '\n' ' ')"
    rm -f "$OUT_DIR/outputs_$STAMP.tar"
  fi
elif [ -d "$OUTPUT_DIR" ]; then
  echo "   outputs not packed ($(du -sh --exclude=_backup "$OUTPUT_DIR" 2>/dev/null | cut -f1) in $OUTPUT_DIR); rerun with INCLUDE_OUTPUTS=1 to include them"
fi

# Without a volume, ComfyUI's own /view endpoint is the easiest way to get the tars off the pod:
# it serves any file in output/ through the same RunPod address you open ComfyUI at.
LINKS=()
if [ "$PERSISTENT" = no ]; then
  mkdir -p "$OUTPUT_DIR/_backup"
  for t in "$TAR:backup.tar" "$OUT_TAR:outputs.tar"; do
    src=${t%%:*}; name=${t##*:}
    [ -n "$src" ] || continue
    rm -f "$OUTPUT_DIR/_backup/$name"
    if ln "$src" "$OUTPUT_DIR/_backup/$name" 2>/dev/null || cp "$src" "$OUTPUT_DIR/_backup/$name"; then
      LINKS+=("$name ($(human "$(stat -c %s "$src")"), $(stat -c %s "$src") bytes): https://${RUNPOD_POD_ID:-YOUR-POD-ID}-$PORT.proxy.runpod.net/view?type=output&subfolder=_backup&filename=$name")
    else
      block "Couldn't put $name in $OUTPUT_DIR/_backup for download."
    fi
  done
  echo
  echo "   Everything on this pod is deleted on Stop. Biggest folders, so you can spot anything else you need:"
  du -sh "$WS_REAL"/* "$WS_REAL"/runpod-slim/* "$COMFY_DIR"/models/* 2>/dev/null | sort -rh | grep -v '_checkpoint' | head -20 | sed 's/^/     /'
fi

# --------------------------------------------------------------------------------------------
step 5/5 "Verdict"
if [ ${#WARNINGS[@]} -gt 0 ]; then
  echo "   Warnings (no data loss, but note them):"
  printf '    - %s\n' "${WARNINGS[@]}"
fi
echo
if [ "$PERSISTENT" = no ]; then
  echo "   >>> THIS POD HAS NO VOLUME. Stop (or a restart) deletes everything on it: ComfyUI, models, LoRAs,"
  echo "       workflows, outputs. Only stop it once the downloads below are saved on your computer."
  if [ ${#BLOCKERS[@]} -gt 0 ]; then
    echo "   Also fix these first, then rerun:"
    printf '    - %s\n' "${BLOCKERS[@]}"
  fi
elif [ ${#BLOCKERS[@]} -gt 0 ]; then
  echo "   >>> NOT SAFE TO STOP YET. Fix these first, then rerun:"
  printf '    - %s\n' "${BLOCKERS[@]}"
else
  echo "   >>> SAFE TO STOP. Use Stop, never Terminate."
fi
if [ ${#LINKS[@]} -gt 0 ]; then
  echo
  echo "   Download these in your browser BEFORE you Stop, and check each file's size matches:"
  printf '     %s\n' "${LINKS[@]}"
  echo "   If a link doesn't open, use the address you open ComfyUI at and keep the /view?... part."
elif [ -n "$TAR" ]; then
  echo
  echo "   Optional insurance: copy the backup off the pod. Run this here, then run the"
  echo "   'runpodctl receive ...' line it prints on your own computer:"
  echo "     runpodctl send $TAR"
fi
echo
echo "   Checkpoint folder: $OUT_DIR"
[ ${#BLOCKERS[@]} -eq 0 ] && [ "$PERSISTENT" = yes ]
