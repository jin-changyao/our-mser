#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="${1:-/home/pc/jcy/MELD_raw}"
if [ -f "${INPUT_PATH}" ]; then
    ARCHIVE="$(readlink -f "${INPUT_PATH}")"
    RAW_BASE="$(dirname "${ARCHIVE}")"
else
    RAW_BASE="$(readlink -m "${INPUT_PATH}")"
    ARCHIVE="${RAW_BASE}/MELD.Raw.tar.gz"
fi
RAW_ROOT="${RAW_BASE}/MELD.Raw"

count_mp4() {
    local count=0
    local path
    for path in "$@"; do
        if [ -d "${path}" ]; then
            count=$((count + $(find "${path}" -type f -name "*.mp4" ! -name "._*" | wc -l)))
        fi
    done
    echo "${count}"
}

extract_inner_if_needed() {
    local archive_name="$1"
    shift
    local already=0
    local path
    for path in "$@"; do
        if [ -d "${path}" ] && [ "$(count_mp4 "${path}")" -gt 0 ]; then
            already=1
        fi
    done
    if [ "${already}" -eq 1 ]; then
        echo "${archive_name} already appears extracted"
        return
    fi
    if [ -f "${archive_name}" ]; then
        echo "Extracting ${archive_name}"
        tar -xzf "${archive_name}"
    else
        echo "Warning: ${archive_name} not found"
    fi
}

echo "MELD raw base: ${RAW_BASE}"
echo "Archive: ${ARCHIVE}"
echo "Raw root: ${RAW_ROOT}"

if [ ! -f "${ARCHIVE}" ] && [ ! -d "${RAW_ROOT}" ]; then
    echo "Neither archive nor extracted directory exists."
    exit 1
fi

if [ ! -d "${RAW_ROOT}" ]; then
    echo "Extracting outer archive"
    mkdir -p "${RAW_BASE}"
    tar -xzf "${ARCHIVE}" -C "${RAW_BASE}"
else
    echo "Outer directory already exists"
fi

if [ ! -d "${RAW_ROOT}" ]; then
    echo "Expected extracted directory not found after outer extraction: ${RAW_ROOT}"
    echo "Top-level contents of ${RAW_BASE}:"
    find "${RAW_BASE}" -maxdepth 2 -type d | sort | head -50
    exit 1
fi

cd "${RAW_ROOT}"

extract_inner_if_needed train.tar.gz train_splits train
extract_inner_if_needed dev.tar.gz dev_splits_complete dev_splits dev
extract_inner_if_needed test.tar.gz output_repeated_splits_test test_splits test

echo "Removing AppleDouble metadata files if present"
find . -name "._*" -delete

echo "CSV files:"
for csv in train_sent_emo.csv dev_sent_emo.csv test_sent_emo.csv; do
    if [ -f "${csv}" ]; then
        ls -lh "${csv}"
    else
        echo "Missing: ${csv}"
    fi
done

echo "Video counts:"
printf "  train: "
count_mp4 train_splits train
printf "  dev:   "
count_mp4 dev_splits_complete dev_splits dev
printf "  test:  "
count_mp4 output_repeated_splits_test test_splits test

echo "Sample videos:"
find . -type f \( -name "dia0_utt0.mp4" -o -name "dia0_utt1.mp4" \) | sort | head -20

echo "MELD raw data is ready at: ${RAW_ROOT}"
