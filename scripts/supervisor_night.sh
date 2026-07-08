#!/bin/bash
# Overnight supervisor for Engram training.
#
# Policy:
#   - Keep --xsa-enabled=1, --engram-enabled=1, --train-engram-only=1 on every launch.
#   - Never remove features. Only adjust stage progression.
#   - On process exit: resume from latest step_NNNNNN ckpt, escalate to next stage.
#   - Stage ladder (max_steps absolute, starting from 6440 Phase 1.6 XSA ckpt):
#        Stage 1: 6740  (10M  tokens, current probe)
#        Stage 2: 9740  (100M tokens continuation)
#        Stage 3: 15740 (500M tokens continuation)
#        Stage 4: done — exit supervisor.
#   - If 3 consecutive launches each die within 60 s: stop supervising (likely
#     config bug; better to leave the GPU idle than thrash).
#   - Logs decisions to runs/phase3_engram_l3_l19/supervisor.log.

set -u

RUN=phase3_engram_l3_l19
DIR=/thearray/git/moe-mla/runs/$RUN
LOG=$DIR/supervisor.log
CONFIG_PATCH=/thearray/git/moe-mla/converted_bkv
ENGRAM_PATCH=/thearray/git/moe-mla/engram_converted_l3_l19
VENV=/thearray/git/moe-mla/.venv

STAGES=(6740 9740 15740)
STAGE_LABELS=("10M probe" "100M continuation" "500M continuation")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

get_latest_step() {
    ls -d "$DIR"/step_[0-9]* 2>/dev/null | sort -V | tail -1 | xargs -n1 -I{} basename {} 2>/dev/null | sed 's/^step_0*//;s/^$/0/'
}

current_max_steps_target() {
    local latest="$1"
    for s in "${STAGES[@]}"; do
        if [ "$latest" -lt "$s" ]; then
            echo "$s"
            return
        fi
    done
    echo "done"
}

stage_label_for_max() {
    local target="$1"
    local i=0
    for s in "${STAGES[@]}"; do
        if [ "$s" = "$target" ]; then
            echo "${STAGE_LABELS[$i]}"
            return
        fi
        i=$((i + 1))
    done
    echo "unknown"
}

launch_training() {
    local resume_ckpt="$1"
    local max_steps="$2"
    local label="$3"
    cd /thearray/git/moe-mla
    log "launching $label: resume=$resume_ckpt max_steps=$max_steps"
    source "$VENV/bin/activate"
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
    # MOE_MLA_ENGRAM_COMPILE=0: torch.compile on Engram's post-embedding path
    # produced ~1% bf16 numerical drift vs eager (verified by direct comparison),
    # and over 14 training steps we saw h1_ppl jump 3.27→3.40 (vs ~0.04 normal
    # variance). ROI wasn't clear in tok/sec either. Keep eager for now; revisit
    # if the training-gradient drift turns out to be unrelated.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MOE_MLA_ENGRAM_COMPILE=0 \
    nohup python -m rwkv_lab.train_mla \
        --resume "$resume_ckpt" \
        --patch-dir "$CONFIG_PATCH" \
        --xsa-enabled 1 \
        --engram-enabled 1 \
        --engram-patch-dir "$ENGRAM_PATCH" \
        --train-engram-only 1 \
        --install-mtp 1 --train-mtp-only 0 --train-aux-only 0 \
        --mtp-chain-horizons "2,3,4" --mtp-chain-weights "1.0,0.5,0.25" \
        --mtp-loss-weight 0.3 \
        --mutor-enabled 1 --mutor-weight 0.1 --mutor-num-registers 32 --mutor-d-max 4 \
        --fsp-enabled 1 --fsp-weight 0.05 --fsp-num-positions 64 --fsp-tau 12 \
        --engram-lr-mult 0.5 \
        --tokens-bin /thearray/data/non_cvevc_tokens.bin \
        --total-tokens-in-bin 29284583603 \
        --max-steps "$max_steps" \
        --warmup-steps 50 --resume-warmup-steps 100 \
        --micro-batch-size 1 --grad-accum-steps 16 \
        --lr 5e-5 --min-lr 5e-6 \
        --log-every 20 --eval-every 25 --eval-batches 8 \
        --save-every 500 --save-every-seconds 1800 --max-saved-checkpoints 3 \
        --out-dir "$DIR" \
        > "$DIR/stdout_$(date +%s).log" 2>&1 &
    disown
    sleep 10
    local pid
    pid=$(pgrep -f "train_mla.py.*$RUN" | head -1)
    if [ -n "$pid" ]; then
        log "launched PID $pid"
        echo "$pid" > "$DIR/.supervisor_child_pid"
        local start_time
        start_time=$(date +%s)
        echo "$start_time" > "$DIR/.supervisor_child_start"
    else
        log "ERROR: launch failed — no python PID found"
    fi
}

main() {
    mkdir -p "$DIR"
    log "==== overnight supervisor starting ===="
    log "run=$RUN  stages=${STAGES[*]}"

    local consec_fails=0

    while true; do
        # Wait for current process (if any) to exit
        while pgrep -f "train_mla.py.*$RUN" >/dev/null; do
            sleep 60
        done
        log "no training process running"

        # Check how long the last run lived (for thrash-detection)
        local ran_for=0
        if [ -f "$DIR/.supervisor_child_start" ]; then
            local start_ts end_ts
            start_ts=$(cat "$DIR/.supervisor_child_start")
            end_ts=$(date +%s)
            ran_for=$((end_ts - start_ts))
        fi

        if [ "$ran_for" -gt 0 ] && [ "$ran_for" -lt 60 ]; then
            consec_fails=$((consec_fails + 1))
            log "run lived only ${ran_for}s — consec_fails=$consec_fails"
        elif [ "$ran_for" -gt 0 ]; then
            consec_fails=0
            log "run lived ${ran_for}s"
        fi

        if [ "$consec_fails" -ge 3 ]; then
            log "3 consecutive quick failures. Stopping supervisor to avoid thrash."
            exit 1
        fi

        # Pick next stage based on latest checkpoint
        local latest
        latest=$(get_latest_step)
        if [ -z "$latest" ] || [ "$latest" = "0" ]; then
            log "ERROR: no step_* ckpt found in $DIR. Cannot resume. Exiting."
            exit 1
        fi
        log "latest ckpt step: $latest"

        local target
        target=$(current_max_steps_target "$latest")
        if [ "$target" = "done" ]; then
            log "all stages complete (step $latest >= final target). Exiting supervisor."
            exit 0
        fi
        local label
        label=$(stage_label_for_max "$target")
        local ckpt_path
        ckpt_path="$DIR/step_$(printf '%06d' "$latest")/ckpt.pt"
        if [ ! -f "$ckpt_path" ]; then
            log "ERROR: ckpt file missing: $ckpt_path"
            exit 1
        fi
        launch_training "$ckpt_path" "$target" "$label"

        # Give the new process time to get going before we re-enter the loop
        sleep 60
    done
}

main
