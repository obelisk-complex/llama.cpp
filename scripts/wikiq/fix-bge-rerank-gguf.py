#!/usr/bin/env python3
"""Rebuild gpustack's bge-reranker-v2-m3-FP16.gguf so llama.cpp assembles the
same query/document pair HF's reference tokenizer does.

One transform, verified against the HF reference (BAAI/bge-reranker-v2-m3,
stock AutoModelForSequenceClassification, no trust_remote_code needed):
logits agree within +/-0.05 with this fix, up to 0.38 without it, on two
independent input sets (rank order matched the reference either way, so
this closes a magnitude gap, not an ordering one - see wikiq's fork commit
9c51923 for the separate token_type crash fix this model also needed).

The GGUF is missing tokenizer.ggml.add_sep_token (absent entirely, not set
false). Without it, llama.cpp's rerank prompt assembly produces
<s> query </s> document </s> - one separator between segments. HF's own
tokenizer (transformers' sentencepiece/XLM-RoBERTa convention, add_sep_token
implied) produces <s> query </s></s> document </s> - two separators, the
second marking the actual segment boundary. Missing that second </s> shifts
every downstream position and drifts the classifier's input from what it
was trained on.

Pure GGUF metadata edit: append one new BOOL key-value pair, bump n_kv, and
copy every tensor through byte-identical (no tensor payload changes, unlike
wikiq's jina-reranker fix script - this one is not touching weights). No
non-stdlib dependencies.

Usage: fix-bge-rerank-gguf.py <src gpustack gguf> <dst gguf>
"""
import struct
import sys

SRC, DST = sys.argv[1], sys.argv[2]
NEW_KEY = b"tokenizer.ggml.add_sep_token"
GGUF_TYPE_BOOL = 7

f = open(SRC, "rb")
assert f.read(4) == b"GGUF"
ver, = struct.unpack("<I", f.read(4))
n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

ALIGN = 32


def u32():
    return struct.unpack("<I", f.read(4))[0]


def u64():
    return struct.unpack("<Q", f.read(8))[0]


def read_str():
    n = u64()
    return f.read(n)


def skip_val(t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes:
        f.read(sizes[t])
    elif t == 8:
        read_str()
    elif t == 9:
        et = u32()
        cnt = u64()
        for _ in range(cnt):
            skip_val(et)
    else:
        raise ValueError("unhandled GGUF value type %d" % t)


# ---- walk existing KV pairs, refuse to touch a file that already has the key
kv_start = f.tell()
for _ in range(n_kv):
    key = read_str()
    ty = u32()
    if key == NEW_KEY:
        raise SystemExit(
            "%s already present in %s; this file does not need the fix" % (NEW_KEY.decode(), SRC)
        )
    skip_val(ty)
kv_end = f.tell()

f.seek(kv_start)
kv_bytes = f.read(kv_end - kv_start)

new_kv = (
    struct.pack("<Q", len(NEW_KEY)) + NEW_KEY
    + struct.pack("<I", GGUF_TYPE_BOOL)
    + struct.pack("<B", 1)
)

# ---- tensor info: read through verbatim, offsets are relative to the
# (realigned) data section start, so they stay correct unchanged
tensor_info_start = f.tell()
for _ in range(n_tensors):
    read_str()          # name
    nd = u32()
    for _ in range(nd):
        u64()            # dim
    u32()                 # ggml type
    u64()                 # offset
tensor_info_end = f.tell()

f.seek(tensor_info_start)
tensor_info_bytes = f.read(tensor_info_end - tensor_info_start)

data_start_old = f.tell()
data_start_old = (data_start_old + ALIGN - 1) // ALIGN * ALIGN
f.seek(data_start_old)
tensor_data = f.read()
f.close()

# ---- write: header, grown KV block, unchanged tensor-info block, realigned
# tensor data (byte-identical payload, only its absolute file position moves)
w = open(DST, "wb")
w.write(b"GGUF")
w.write(struct.pack("<IQQ", ver, n_tensors, n_kv + 1))
w.write(kv_bytes)
w.write(new_kv)
w.write(tensor_info_bytes)

pos = w.tell()
pad = (pos + ALIGN - 1) // ALIGN * ALIGN - pos
w.write(b"\x00" * pad)
w.write(tensor_data)
w.close()

print("wrote", DST, "-", n_tensors, "tensors, added", NEW_KEY.decode())
