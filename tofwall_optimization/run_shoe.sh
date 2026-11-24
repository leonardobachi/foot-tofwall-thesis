#!/bin/bash
# run_shoe.sh — SHOE reconstruction (DecodeGlb) of a list of runs, several runs in parallel.
#
# To be launched from the SHOE folder build/Reconstruction (DecodeGlb in ../bin).
# For each run, DecodeGlb starts from the first .data file of <data_dir>/<run>/
# and reads the following files of the same folder by itself; the output is
# <out_dir>/<run>.root, the log <out_dir>/logs/<run>.log. Runs with an existing
# .root are skipped.
#
# Usage:
#   ./run_shoe.sh <data_dir> <out_dir> [n_parallel]       (default n_parallel = 4)
#
# Variants:
#   - campaign and runs: EXP and RUNS below
#   - FILTER_LOG=1 removes the very verbose per-cluster lines of the vertex/inner
#     tracker from the logs (runs without magnetic field)
#   - 2025 .bin files (raw WaveDAQ): use ../bin/DecodeWD instead of DecodeGlb, with
#     the same options, looping on "$DATA_DIR"/*.bin
#   - one .root per .data file (e.g. TW scans): move each .data into its own
#     sub-folder before running, otherwise DecodeGlb chains all the files of the folder
# For the TW hits to be saved in the .root, EnableSaveHits must be "y" in
# config/<EXP>/FootGlobal.par.

EXP=CNAO2025

# runs with magnetic field
RUNS=(8101 8102 8103 8105 8106 8107 8109 8110 8111 8112 8118 8119 8120 8121 8122 8124
      8125 8126 8127 8128 8130 8131 8132 8133 8135 8136 8137 8139 8140 8141 8142 8143
      8144 8147 8148 8149 8150 8153 8154 8155 8156 8159 8160 8161 8162 8164)
# runs without magnetic field (use with FILTER_LOG=1)
# RUNS=(7908 7909 7910 7911 7922 7924 7927 7928 7930 7931 7932 7933 7940 7941 7942 7943
#       8028 8039 8040 8093 8094 8096 8097 8098 8099)

FILTER_LOG=0

BASE_DATA_DIR=$1
BASE_OUT_DIR=$2
N_PARALLEL=${3:-4}

if [ -z "$BASE_DATA_DIR" ] || [ -z "$BASE_OUT_DIR" ]; then
    echo "Usage: ./run_shoe.sh <data_dir> <out_dir> [n_parallel]"
    exit 1
fi

LOG_DIR="$BASE_OUT_DIR/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$BASE_OUT_DIR/run_shoe.log"

echo "Start: $(date)" | tee -a "$MAIN_LOG"
echo "Experiment: $EXP, ${#RUNS[@]} runs, $N_PARALLEL in parallel" | tee -a "$MAIN_LOG"

process_run() {
    RUN=$1
    DATA_DIR="${BASE_DATA_DIR}/${RUN}"
    OUT_ROOT="${BASE_OUT_DIR}/${RUN}.root"
    RUN_LOG="${LOG_DIR}/${RUN}.log"

    if [ -f "$OUT_ROOT" ]; then
        echo "[SKIP] Run $RUN — $OUT_ROOT already exists"
        return 0
    fi

    first_file=$(ls "$DATA_DIR"/*.data 2>/dev/null | sort | head -n 1)
    if [ -z "$first_file" ]; then
        echo "[ERR] Run $RUN — no .data file in $DATA_DIR"
        return 1
    fi

    echo ">>> Run $RUN — $(date +%H:%M:%S) — starting from $first_file" > "$RUN_LOG"

    if [ "$FILTER_LOG" = "1" ]; then
        ../bin/DecodeGlb -in "$first_file" -exp "$EXP" -run "$RUN" -out "$OUT_ROOT" 2>&1 \
            | grep -vE '^(Cluster idx|Strip:|position of cluster|pos error of cluster|tot energy loss of cluster|sensor idx)' \
            >> "$RUN_LOG"
    else
        ../bin/DecodeGlb -in "$first_file" -exp "$EXP" -run "$RUN" -out "$OUT_ROOT" >> "$RUN_LOG" 2>&1
    fi

    if [ -f "$OUT_ROOT" ]; then
        echo "[OK] Run $RUN — $(du -h "$OUT_ROOT" | cut -f1)"
    else
        echo "[ERR] Run $RUN — $OUT_ROOT not created, see $RUN_LOG"
    fi
}
export -f process_run
export BASE_DATA_DIR BASE_OUT_DIR LOG_DIR EXP FILTER_LOG

printf "%s\n" "${RUNS[@]}" | xargs -I{} -P "$N_PARALLEL" bash -c 'process_run {}' | tee -a "$MAIN_LOG"

# per-run logs appended to the main log
echo "" >> "$MAIN_LOG"
echo "=== Per-run logs ===" >> "$MAIN_LOG"
for RUN in "${RUNS[@]}"; do
    if [ -f "${LOG_DIR}/${RUN}.log" ]; then
        echo "--- Run $RUN ---" >> "$MAIN_LOG"
        cat "${LOG_DIR}/${RUN}.log" >> "$MAIN_LOG"
    fi
done

echo "End: $(date)" | tee -a "$MAIN_LOG"
