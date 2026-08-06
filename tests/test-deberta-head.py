# Converts the synthetic checkpoint (zero encoder + the hand-chosen 3-class head
# from Task 5) and asserts the RANK-pooled logits equal the gelu-erf reference.
# The encoder is all zeros, so hidden states are exactly zero, the pooler dense
# reduces to its bias POOLER_BIAS, and:
#     logits = [1, 2, -1] * act(POOLER_BIAS)
# gelu_erf(2.0) = 1.9544997 vs tanh(2.0) = 0.9640276, so a wrong activation
# fails by ~0.99 per logit, far outside TOL.
import importlib.util, math, os, pathlib, subprocess, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parent.parent
TOL = 1e-3
# ctest passes LLAMA_EMBEDDING_BIN; the default is the conventional build dir.
EMB = pathlib.Path(os.environ.get("LLAMA_EMBEDDING_BIN",
                                 str(ROOT / "build" / "bin" / "llama-embedding")))

def load_builder():
    spec = importlib.util.spec_from_file_location("conv", ROOT / "tests" / "test-deberta-convert.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def gelu_erf(x: float) -> float:
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def main():
    m = load_builder()
    act = gelu_erf(m.POOLER_BIAS)                      # 1.9544997361036416 at 2.0
    want = [1.0 * act, 2.0 * act, -1.0 * act]          # classifier selector rows
    wrong = math.tanh(m.POOLER_BIAS)                   # 0.9640275800758169
    with tempfile.TemporaryDirectory() as td:
        dm = pathlib.Path(td) / "m"; dm.mkdir(); m.build_synthetic(dm)
        gguf = pathlib.Path(td) / "m.gguf"
        subprocess.run([sys.executable, str(ROOT / "convert_hf_to_gguf.py"), str(dm),
                        "--outfile", str(gguf), "--outtype", "f32"], check=True)
        # --embd-normalize -1: keep raw logits. The default (2, euclidean) would
        # rescale the vector and destroy the comparison against `want`.
        # --embd-output-format raw: one whitespace-delimited line of exactly
        # n_cls_out values at %1.7f. The default RANK format is prose at only
        # three decimal places, too coarse for TOL.
        out = subprocess.run([str(EMB),
                              "-m", str(gguf), "-p", "premise hypothesis",
                              "--pooling", "rank", "--embd-normalize", "-1",
                              "--embd-output-format", "raw"],
                             capture_output=True, text=True, check=True).stdout
        # Strict parse. llama.cpp sends only LOG() (level NONE) to stdout and
        # every other log level to stderr, so stdout is the embedding line alone.
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert len(lines) == 1, (
            f"expected exactly one raw embedding line, got {len(lines)}: {lines!r}\n"
            f"--- full stdout ---\n{out}")
        try:
            vals = [float(t) for t in lines[0].split()]
        except ValueError as e:
            raise AssertionError(
                f"non-float token in embedding line {lines[0]!r}: {e}\n"
                f"--- full stdout ---\n{out}")
        assert len(vals) == 3, (
            f"RANK-pooled head must emit 3 logits, got {len(vals)} in {lines[0]!r}\n"
            f"--- full stdout ---\n{out}")
        assert all(v == v for v in vals), f"NaN in head output: {vals}"
        for i, (g, w) in enumerate(zip(vals, want)):
            assert abs(g - w) <= TOL, (
                f"logit {i}: got {g}, expected {w} (gelu-erf pooler). "
                f"tanh would give {[1.0, 2.0, -1.0][i] * wrong}; a hardcoded tanh "
                f"in build_pooling's RANK path is the usual cause.")
        print("test-deberta-head: OK")

if __name__ == "__main__":
    main()
