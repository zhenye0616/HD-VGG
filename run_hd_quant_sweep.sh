#!/usr/bin/env bash

# Sweep hd_dim values for VGG + HD head using existing main.py entrypoint.
# Each run enforces 4-bit fake weight quantization and 5-bit fake activation quantization.
# Results are streamed to stdout, appended to logs/hd_quant_runs.log,
# and a compact per-run summary with the final accuracy is written to logs/hd_quant_summary.txt.

set -euo pipefail

HD_DIMS=(8 32 64 128 256 512 1024)
LOG_DIR="logs"
FULL_LOG="${LOG_DIR}/hd_quant_runs.log"
SUMMARY_LOG="${LOG_DIR}/hd_quant_summary.txt"

mkdir -p "${LOG_DIR}"

echo "Starting HD sweep: hd_dims=${HD_DIMS[*]}, weight_bits=4, activation_bits=5" | tee -a "${FULL_LOG}"

for hd_dim in "${HD_DIMS[@]}"; do
    echo "------------------------------------------------------------------" | tee -a "${FULL_LOG}"
    echo "Running hd_dim=${hd_dim}..." | tee -a "${FULL_LOG}"

    tmp_log="$(mktemp)"
    if python3 main.py \
        --use_hd_classifier \
        --hd_dim "${hd_dim}" \
        --network_quantization \
        --network_quantization_bits 4 \
        --data_quantization \
        --data_quantization_bits 5 \
        "$@" | tee "${tmp_log}"; then

        cat "${tmp_log}" >> "${FULL_LOG}"
        acc_line="$(grep -E "\[Epoch .*Test  Acc" "${tmp_log}" | tail -n 1 || true)"
        rm -f "${tmp_log}"

        if [[ -n "${acc_line}" ]]; then
            acc_value="$(echo "${acc_line}" | awk -F'Test  Acc: ' '{print $2}')"
        else
            acc_value="Test  Acc: N/A"
        fi

        summary_line="$(date -Iseconds) hd_dim=${hd_dim} | w_bits=4 | act_bits=5 | ${acc_value}"
        echo "${summary_line}" | tee -a "${SUMMARY_LOG}"
    else
        status=$?
        cat "${tmp_log}" >> "${FULL_LOG}"
        rm -f "${tmp_log}"
        echo "$(date -Iseconds) hd_dim=${hd_dim} FAILED (exit ${status})" | tee -a "${SUMMARY_LOG}"
        exit "${status}"
    fi
done

echo "Sweep complete. Full log: ${FULL_LOG}, summaries: ${SUMMARY_LOG}" | tee -a "${FULL_LOG}"
