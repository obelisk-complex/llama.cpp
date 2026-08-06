#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
# ctest passes LLAMA_PYTHON=${Python3_EXECUTABLE}, the same interpreter Tasks 5
# and 11 register with and the one Task 5's prerequisites are installed into.
# The inner convert_hf_to_gguf.py call inherits it through sys.executable.
py="${LLAMA_PYTHON:-python3}"
"${py}" - "$ROOT" "$work" <<'PY'
import importlib.util, pathlib, subprocess, sys
root, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("conv", root / "tests" / "test-deberta-convert.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dm = work / "m"; dm.mkdir(); m.build_synthetic(dm)
subprocess.run([sys.executable, str(root / "convert_hf_to_gguf.py"), str(dm),
                "--outfile", str(work / "m.gguf"), "--outtype", "f32"], check=True)
PY
# ctest passes LLAMA_EMBEDDING_BIN; the default is the conventional build dir.
emb="${LLAMA_EMBEDDING_BIN:-${ROOT}/build/bin/llama-embedding}"
# stderr is NOT redirected: it carries the load/encode error that explains a
# real failure, and set -e already fails the script on a non-zero exit.
out="$("${emb}" -m "${work}/m.gguf" -p "premise hypothesis" --pooling none)"
[ -n "${out}" ] || { echo "llama-embedding produced no embedding output" >&2; exit 1; }
if echo "${out}" | grep -qiE 'nan|inf'; then
  echo "non-finite value in embedding output:" >&2
  echo "${out}" >&2
  exit 1
fi
echo "test-deberta-encode: OK"
