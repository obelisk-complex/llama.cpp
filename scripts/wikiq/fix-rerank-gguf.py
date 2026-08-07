#!/usr/bin/env python3
"""Rebuild ggml-org's Jina-Bert-Implementation-38M-F16.gguf so llama.cpp
reproduces the HF reference forward pass of jinaai/jina-reranker-v1-turbo-en.

Two transforms, both required (verified against the HF reference, revision
b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2, logits agree within +/-0.005):

1. Classification head. The ggml-org GGUF omits bert.pooler.dense.{weight,bias}
   and stores HF's classifier (Linear 384->1) as `cls`. HF computes
   classifier(tanh(pooler_dense(CLS))); llama.cpp's RANK pooling computes
   cls -> tanh -> cls.output. So: cls := pooler.dense (384x384, F32),
   cls.output := classifier (384->1, F32), both pulled from the HF
   safetensors via HTTP range requests.

2. ALiBi head padding. Jina's modeling_bert.py halves the interpolated ALiBi
   slopes for heads 8-11 ("quick fix on large jump at header=12"):
     jina slopes = [2^-1 .. 2^-8, 2^-1.5, 2^-2.5, 2^-3.5, 2^-4.5]
   ggml hardcodes the paper-standard interpolation, but all 12 jina slopes are
   members of ggml's 16-head slope set 2^-0.5(p+1). Pad attention to 16 heads
   (zeroed Q/K/V rows and wo columns for the 4 pad heads) and permute head
   blocks so jina head j sits at slope-matching position p:
     j:  0  1  2  3  4   5   6   7   8  9  10  11
     p:  1  3  5  7  9  11  13  15   2  4   6   8    (0, 10, 12, 14 zeroed)
   Metadata: attention.head_count 12->16, attention.key_length/value_length 32.
   Requires the fork's jina-bert-v2 loader change (q/wo dims from
   n_embd_head_k * n_head instead of n_embd).

Usage: fix-rerank-gguf.py <src ggml-org gguf> <dst gguf>
"""
import json
import struct
import sys
import urllib.request

ST_URL = ('https://huggingface.co/jinaai/jina-reranker-v1-turbo-en/resolve/'
          'b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2/model.safetensors')
ALIGN = 32
HEAD_DIM = 32
N_HEAD_NEW = 16
P_OF_J = [1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8]
J_OF_P = {p: j for j, p in enumerate(P_OF_J)}

SRC, DST = sys.argv[1], sys.argv[2]


# ---- fetch head tensors from the HF safetensors ------------------------------
def http_range(url, a, b):
    req = urllib.request.Request(url, headers={'Range': 'bytes=%d-%d' % (a, b)})
    return urllib.request.urlopen(req).read()

head = http_range(ST_URL, 0, 131071)
hlen = struct.unpack('<Q', head[:8])[0]
st_hdr = json.loads(head[8:8 + hlen])
st_base = 8 + hlen

def fetch_bf16_as_f32(name, shape):
    info = st_hdr[name]
    assert info['dtype'] == 'BF16' and info['shape'] == shape, (name, info)
    a, b = info['data_offsets']
    raw = http_range(ST_URL, st_base + a, st_base + b - 1)
    n = (b - a) // 2
    u16 = struct.unpack('<%dH' % n, raw)
    return struct.pack('<%dI' % n, *[v << 16 for v in u16])

pool_w = fetch_bf16_as_f32('bert.pooler.dense.weight', [384, 384])
pool_b = fetch_bf16_as_f32('bert.pooler.dense.bias', [384])
clf_w = fetch_bf16_as_f32('classifier.weight', [1, 384])
clf_b = fetch_bf16_as_f32('classifier.bias', [1])


# ---- parse source GGUF -------------------------------------------------------
f = open(SRC, 'rb')
assert f.read(4) == b'GGUF'
ver, = struct.unpack('<I', f.read(4))
n_tensors, n_kv = struct.unpack('<QQ', f.read(16))

def u32(): return struct.unpack('<I', f.read(4))[0]
def u64(): return struct.unpack('<Q', f.read(8))[0]
def s():
    n = u64()
    return f.read(n)

def skip_val(t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes:
        f.read(sizes[t])
    elif t == 8:
        s()
    elif t == 9:
        et = u32(); cnt = u64()
        for _ in range(cnt):
            skip_val(et)
    else:
        raise Exception('bad kv type %d' % t)

kv_start = f.tell()
head_count_val_off = None
for _ in range(n_kv):
    key = s()
    ty = u32()
    if key == b'jina-bert-v2.attention.head_count':
        assert ty == 4
        head_count_val_off = f.tell()
    skip_val(ty)
kv_end = f.tell()
assert head_count_val_off is not None

f.seek(kv_start)
kv_bytes = bytearray(f.read(kv_end - kv_start))
rel = head_count_val_off - kv_start
n_head_old = struct.unpack_from('<I', kv_bytes, rel)[0]
assert n_head_old == 12, n_head_old
struct.pack_into('<I', kv_bytes, rel, N_HEAD_NEW)

def kv_u32(key, val):
    k = key.encode()
    return struct.pack('<Q', len(k)) + k + struct.pack('<II', 4, val)

extra_kv = kv_u32('jina-bert-v2.attention.key_length', HEAD_DIM) + \
           kv_u32('jina-bert-v2.attention.value_length', HEAD_DIM)

infos = []
for _ in range(n_tensors):
    name = s(); nd = u32()
    dims = [u64() for _ in range(nd)]
    ty = u32(); off = u64()
    infos.append([name, dims, ty, off])
info_end = f.tell()
data_start = (info_end + ALIGN - 1) // ALIGN * ALIGN

TYPE_BYTES = {0: 4, 1: 2}
def nbytes(dims, ty):
    n = 1
    for dd in dims:
        n *= dd
    return n * TYPE_BYTES[ty]


# ---- transforms --------------------------------------------------------------
def permute_out_blocks(data, in_dim, elt):
    row = in_dim * elt
    blk = HEAD_DIM * row
    out = bytearray(N_HEAD_NEW * blk)
    for p in range(N_HEAD_NEW):
        j = J_OF_P.get(p)
        if j is not None:
            out[p*blk:(p+1)*blk] = data[j*blk:(j+1)*blk]
    return bytes(out)

def permute_in_blocks(data, out_dim, in_dim, elt):
    old_row = in_dim * elt
    new_row = N_HEAD_NEW * HEAD_DIM * elt
    blk = HEAD_DIM * elt
    out = bytearray(out_dim * new_row)
    for r in range(out_dim):
        src = data[r*old_row:(r+1)*old_row]
        base = r * new_row
        for p in range(N_HEAD_NEW):
            j = J_OF_P.get(p)
            if j is not None:
                out[base + p*blk: base + (p+1)*blk] = src[j*blk:(j+1)*blk]
    return bytes(out)

tensors = []
n_attn, n_head_fix = 0, 0
for name, dims, ty, off in infos:
    f.seek(data_start + off)
    data = f.read(nbytes(dims, ty))
    sname = name.decode()
    if sname == 'cls.weight':
        assert dims == [384] and ty == 0
        tensors.append([b'cls.weight', [384, 384], 0, pool_w])
        tensors.append([b'cls.output.weight', [384], 0, clf_w])
        n_head_fix += 1
        continue
    if sname == 'cls.bias':
        assert dims == [1] and ty == 0
        tensors.append([b'cls.bias', [384], 0, pool_b])
        tensors.append([b'cls.output.bias', [1], 0, clf_b])
        n_head_fix += 1
        continue
    if ('.attn_q.weight' in sname or '.attn_k.weight' in sname or '.attn_v.weight' in sname):
        assert dims == [384, 384] and ty == 1
        data = permute_out_blocks(data, 384, 2)
        dims = [384, N_HEAD_NEW * HEAD_DIM]
        n_attn += 1
    elif ('.attn_q.bias' in sname or '.attn_k.bias' in sname or '.attn_v.bias' in sname):
        assert dims == [384] and ty == 0
        data = permute_out_blocks(data, 1, 4)
        dims = [N_HEAD_NEW * HEAD_DIM]
        n_attn += 1
    elif '.attn_output.weight' in sname:
        assert dims == [384, 384] and ty == 1
        data = permute_in_blocks(data, 384, 384, 2)
        dims = [N_HEAD_NEW * HEAD_DIM, 384]
        n_attn += 1
    tensors.append([name, dims, ty, data])
f.close()
assert n_head_fix == 2 and n_attn == 6 * 7, (n_head_fix, n_attn)


# ---- write -------------------------------------------------------------------
w = open(DST, 'wb')
w.write(b'GGUF')
w.write(struct.pack('<IQQ', ver, len(tensors), n_kv + 2))
w.write(kv_bytes)
w.write(extra_kv)

off = 0
enc = b''
offsets = []
for name, dims, ty, data in tensors:
    off = (off + ALIGN - 1) // ALIGN * ALIGN
    offsets.append(off)
    enc += struct.pack('<Q', len(name)) + name
    enc += struct.pack('<I', len(dims))
    for dd in dims:
        enc += struct.pack('<Q', dd)
    enc += struct.pack('<IQ', ty, off)
    off += len(data)
w.write(enc)

header_len = 4 + 4 + 8 + 8
dstart = (header_len + len(kv_bytes) + len(extra_kv) + len(enc) + ALIGN - 1) // ALIGN * ALIGN
assert w.tell() == header_len + len(kv_bytes) + len(extra_kv) + len(enc)
w.write(b'\x00' * (dstart - w.tell()))
for (name, dims, ty, data), o in zip(tensors, offsets):
    w.write(b'\x00' * (dstart + o - w.tell()))
    w.write(data)
w.close()
print('wrote', DST, '-', len(tensors), 'tensors')
