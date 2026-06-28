#!/usr/bin/env bash
# Convert ConDec-BPMN-Modeler JSON models to XES event logs via MINERful.
#
# Usage: bash scripts/generate_xes_from_condec.sh
#
# Reads from:
#   "Synthetic Log Data/looselyStructuredModels/"
#   "Synthetic Log Data/semiStructuredModels/"
# Writes to:
#   "Synthetic Log Data/looselyStructuredLogs/"
#   "Synthetic Log Data/semiStructuredLogs/"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MINERFUL_DIR="$HOME/dev/kerstin-tmp/MINERful"
CONDEC_CONVERTER="$MINERFUL_DIR/condec_to_minerful.py"

JAVA=/usr/lib/jvm/java-21-openjdk/bin/java
CP="$MINERFUL_DIR/MINERful.jar:$MINERFUL_DIR/bin:$(ls "$MINERFUL_DIR/lib/"*.jar | tr '\n' ':')"

TMPDIR_SPECS="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_SPECS"' EXIT

run_minerful() {
    local spec_json="$1"
    local out_xes="$2"
    local minlen="$3"
    local maxlen="$4"

    "$JAVA" -Xmx4G -cp "$CP" minerful.MinerFulLogMakerStarter \
        -iSF "$spec_json" \
        -oLF "$out_xes" \
        -oLL 100 \
        -oLm "$minlen" \
        -oLM "$maxlen" \
        -d none
}

convert_dir() {
    local model_dir="$1"
    local log_dir="$2"
    local minlen="$3"
    local maxlen="$4"

    mkdir -p "$log_dir"

    for json_file in "$model_dir"/*.json; do
        [ -f "$json_file" ] || continue
        base="$(basename "$json_file" .json)"
        spec_tmp="$TMPDIR_SPECS/${base}_spec.json"
        out_xes="$log_dir/${base}.xes"

        echo "[convert] $json_file -> $spec_tmp"
        python3 "$CONDEC_CONVERTER" "$json_file" "$spec_tmp" --name "$base"

        echo "[minerful] $spec_tmp -> $out_xes"
        run_minerful "$spec_tmp" "$out_xes" "$minlen" "$maxlen"

        echo "[done] $out_xes"
    done
}

# Loosely structured: few constraints, wide trace length range
convert_dir \
    "$REPO_ROOT/Synthetic Log Data/looselyStructuredModels" \
    "$REPO_ROOT/Synthetic Log Data/looselyStructuredLogs" \
    2 20

# Semi-structured: moderate constraints, medium trace length range
convert_dir \
    "$REPO_ROOT/Synthetic Log Data/semiStructuredModels" \
    "$REPO_ROOT/Synthetic Log Data/semiStructuredLogs" \
    3 15

echo "All XES logs generated."
