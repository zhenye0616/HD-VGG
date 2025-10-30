#!/usr/bin/env bash

# Evaluate the standard VGG11 classifier (no HD head) with 4-bit fake weight
# quantization and 5-bit fake activation quantization using main.py.
# Captures the full console output and the final accuracy for quick comparison.

set -euo pipefail

LOG_DIR="logs"
FULL_LOG="${LOG_DIR}/vgg_quant_runs.log"
SUMMARY_LOG="${LOG_DIR}/vgg_quant_summary.txt"

mkdir -p "${LOG_DIR}"

echo "------------------------------------------------------------------" | tee -a "${FULL_LOG}"
echo "Running plain VGG11 with 4-bit weights + 5-bit activations..." | tee -a "${FULL_LOG}"

tmp_log="$(mktemp)"
if python3 main.py \
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

    summary_line="$(date -Iseconds) plain_vgg | w_bits=4 | act_bits=5 | ${acc_value}"
    echo "${summary_line}" | tee -a "${SUMMARY_LOG}"
else
    status=$?
    cat "${tmp_log}" >> "${FULL_LOG}"
    rm -f "${tmp_log}"
    echo "$(date -Iseconds) plain_vgg FAILED (exit ${status})" | tee -a "${SUMMARY_LOG}"
    exit "${status}"
fi

echo "Run complete. Full log: ${FULL_LOG}, summaries: ${SUMMARY_LOG}" | tee -a "${FULL_LOG}"
