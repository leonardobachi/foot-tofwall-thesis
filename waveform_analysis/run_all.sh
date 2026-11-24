#!/bin/bash
# run_all.sh — Run analyze.py on all the runs of a data folder, one after the other.
#
# Every numeric sub-folder of <data_dir> is a run (e.g. data/8102/*.bin).
# Runs that already have results_<run>.pkl in <out_dir> are skipped, so an
# interrupted processing can be resumed. Output and errors go to the screen
# and to <out_dir>/run_all.log.
#
# Usage:
#   bash run_all.sh <data_dir> <out_dir> <template_tw.pkl> [analyze.py options]
#
# Examples:
#   bash run_all.sh data/ results/ template_tw.pkl --saturation 200 --cores 8
#   nohup bash run_all.sh data/ results/ template_tw.pkl --cores 8 > nohup.log 2>&1 &
#
# The TW template is built once with
#   python3 analyze.py data/8102 results/ --run 8102 --save-template template_tw.pkl

set -e
set -o pipefail   # the exit code of "cmd | tee" is the one of cmd

DATA_DIR="$1"
OUT_DIR="$2"
TEMPLATE="$3"
EXTRA_ARGS=("${@:4}")

if [ -z "$DATA_DIR" ] || [ -z "$OUT_DIR" ] || [ -z "$TEMPLATE" ]; then
    echo "Usage: bash run_all.sh <data_dir> <out_dir> <template_tw.pkl> [analyze.py options]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run_all.log"

echo "==============================================" | tee -a "$LOG_FILE"
echo "Start: $(date)"                                | tee -a "$LOG_FILE"
echo "Data dir:   $DATA_DIR"                         | tee -a "$LOG_FILE"
echo "Output dir: $OUT_DIR"                          | tee -a "$LOG_FILE"
echo "Template:   $TEMPLATE"                         | tee -a "$LOG_FILE"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Options:    ${EXTRA_ARGS[*]}"              | tee -a "$LOG_FILE"
fi
echo "==============================================" | tee -a "$LOG_FILE"

N_OK=0
N_ERR=0
N_SKIP=0

for RUN_DIR in "$DATA_DIR"/*/; do
    RUN_ID=$(basename "$RUN_DIR")

    if ! [[ "$RUN_ID" =~ ^[0-9]+$ ]]; then
        continue
    fi

    OUT_FILE="$OUT_DIR/results_${RUN_ID}.pkl"
    if [ -f "$OUT_FILE" ]; then
        echo "[SKIP] Run $RUN_ID — already processed ($OUT_FILE)" | tee -a "$LOG_FILE"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    echo ""                                   | tee -a "$LOG_FILE"
    echo ">>> Run $RUN_ID — $(date +%H:%M:%S)" | tee -a "$LOG_FILE"

    CMD=(
        python3 "$SCRIPT_DIR/analyze.py"
        "$RUN_DIR"
        "$OUT_DIR"
        --run "$RUN_ID"
        --load-template "$TEMPLATE"
        "${EXTRA_ARGS[@]}"
    )

    if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        N_OK=$((N_OK + 1))
        echo "[OK] Run $RUN_ID" | tee -a "$LOG_FILE"
    else
        N_ERR=$((N_ERR + 1))
        echo "[ERR] Run $RUN_ID — see the log" | tee -a "$LOG_FILE"
    fi
done

echo ""                                                  | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"
echo "End: $(date)"                                   | tee -a "$LOG_FILE"
echo "OK: $N_OK  |  Errors: $N_ERR  |  Skipped: $N_SKIP" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"
