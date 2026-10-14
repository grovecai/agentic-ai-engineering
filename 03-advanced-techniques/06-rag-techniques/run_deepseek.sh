#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-pipeline}" in
  pipeline)
    script_name="03_rag_pipeline_deepseek.py"
    ;;
  agentic)
    script_name="04_agentic_rag_deepseek.py"
    ;;
  *)
    echo "Usage: $0 [pipeline|agentic]" >&2
    exit 2
    ;;
esac

uv run --directory "$script_dir" python "$script_name"
