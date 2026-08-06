# Builds a 2-layer synthetic DeBERTa-v3 checkpoint (weights + a real tiny
# spm.model + a 3-class head), converts it, and asserts the complete tensor set
# and metadata the loader relies on. No network.
import json, pathlib, subprocess, sys, tempfile
import numpy as np
from safetensors.numpy import save_file
import sentencepiece as spm

ROOT = pathlib.Path(__file__).resolve().parent.parent
# gguf-py is not installed; convert_hf_to_gguf.py:15 puts it on sys.path for the
# subprocess only, so this test must do the same for itself. Without it the
# import below raises ImportError before anything under test runs, and Step 2's
# red is indistinguishable from red for the wrong reason.
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf  # noqa: E402

# Pooler-dense bias for the synthetic head, so the pooler activation is
# evaluated at exactly this input. Chosen because gelu-erf and tanh are far
# apart here: gelu_erf(2.0) = 1.9544997, tanh(2.0) = 0.9640276 (gap ~0.99).
POOLER_BIAS = 2.0

def train_spm(dir_model: pathlib.Path, vocab_size: int) -> int:
    corpus = dir_model / "corpus.txt"
    corpus.write_text(("the quick brown fox jumps over the lazy dog\n"
                       "a premise entails a hypothesis about facts\n") * 200)
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(dir_model / "spm"),
        vocab_size=vocab_size, model_type="unigram", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3)
    sp = spm.SentencePieceProcessor(); sp.load(str(dir_model / "spm.model"))
    return sp.get_piece_size()

def build_synthetic(dir_model: pathlib.Path):
    H, L, Hd, FF, PB, NCLS = 8, 2, 2, 16, 4, 3  # att_span=4 -> rel rows=8
    # The real checkpoint's config vocab_size (128100) exceeds its spm piece
    # count, so set_vocab fills the tail with [PAD{i}] entries and sizes the
    # embedding matrix to the config value, not the spm one. Mirror that here:
    # with V_SPM == V the filler loop never runs and that path ships untested.
    # 64 (the brief's original figure) overflows what unigram EM can seed from
    # this two-sentence corpus in this sentencepiece build ("Vocabulary size
    # too high (64). Please set it to a value <= 36."); 32 is comfortably
    # under that ceiling and still exercises the V < config-vocab_size filler
    # path below.
    V_SPM = train_spm(dir_model, vocab_size=32)
    V = V_SPM + 4
    # The real checkpoint's added_tokens.json is {"[MASK]": 128000}: a single
    # token at the FIRST filler id, with the rest of the tail left as [PAD{i}].
    # Mirror that shape exactly - MASK_ID at V_SPM, tail at V-1 still [PAD] - so
    # this fixture gates the added-tokens block and Task 5's filler loop at once.
    # Without an added_tokens.json here the round-trip cannot see the block's
    # omission, which is how that defect survived to be found against the real
    # checkpoint rather than against this test.
    MASK_ID = V_SPM
    (dir_model / "added_tokens.json").write_text(json.dumps({"[MASK]": MASK_ID}))
    # The real checkpoint's tokenizer_config.json names the same token again in
    # added_tokens_decoder, with special: true. That is what makes [MASK] land
    # as CONTROL rather than the USER_DEFINED the added_tokens.json block above
    # assigns it, and it is the only reason the decoder block is worth porting:
    # without this file here the round-trip cannot tell the two blocks apart.
    (dir_model / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": {
            "0": {"content": "[PAD]", "special": True},
            "1": {"content": "[UNK]", "special": True},
            "2": {"content": "[CLS]", "special": True},
            "3": {"content": "[SEP]", "special": True},
            str(MASK_ID): {"content": "[MASK]", "special": True},
        }
    }))
    cfg = {
        "model_type": "deberta-v2", "architectures": ["DebertaV2ForSequenceClassification"],
        "hidden_size": H, "num_hidden_layers": L, "num_attention_heads": Hd,
        "intermediate_size": FF, "vocab_size": V, "type_vocab_size": 0,
        "max_position_embeddings": 16, "relative_attention": True,
        "position_buckets": PB, "max_relative_positions": -1,
        "pos_att_type": ["p2c", "c2p"], "position_biased_input": False,
        "norm_rel_ebd": "layer_norm", "share_att_key": True,
        "layer_norm_eps": 1e-7, "hidden_act": "gelu", "pooler_hidden_act": "gelu",
        "pooler_hidden_size": H, "pad_token_id": 0,
        "id2label": {"0": "contradiction", "1": "entailment", "2": "neutral"},
    }
    (dir_model / "config.json").write_text(json.dumps(cfg))
    t, r = {}, lambda *s: np.zeros(s, dtype=np.float32)
    t["deberta.embeddings.word_embeddings.weight"] = r(V, H)
    t["deberta.embeddings.LayerNorm.weight"] = r(H); t["deberta.embeddings.LayerNorm.bias"] = r(H)
    # The real checkpoint's 203rd tensor: a persistent I64 buffer, not a weight,
    # with no GGUF counterpart. Present here so the round-trip gates the
    # converter's filter_tensors skip instead of stepping around it.
    t["deberta.embeddings.position_ids"] = np.arange(
        cfg["max_position_embeddings"], dtype=np.int64).reshape(1, -1)
    t["deberta.encoder.rel_embeddings.weight"] = r(2 * PB, H)
    t["deberta.encoder.LayerNorm.weight"] = r(H); t["deberta.encoder.LayerNorm.bias"] = r(H)
    for i in range(L):
        p = f"deberta.encoder.layer.{i}."
        for proj in ("query_proj", "key_proj", "value_proj"):
            t[p + f"attention.self.{proj}.weight"] = r(H, H); t[p + f"attention.self.{proj}.bias"] = r(H)
        t[p + "attention.output.dense.weight"] = r(H, H); t[p + "attention.output.dense.bias"] = r(H)
        t[p + "attention.output.LayerNorm.weight"] = r(H); t[p + "attention.output.LayerNorm.bias"] = r(H)
        t[p + "intermediate.dense.weight"] = r(FF, H); t[p + "intermediate.dense.bias"] = r(FF)
        t[p + "output.dense.weight"] = r(H, FF); t[p + "output.dense.bias"] = r(H)
        t[p + "output.LayerNorm.weight"] = r(H); t[p + "output.LayerNorm.bias"] = r(H)
    # Head: hand-chosen non-zero weights. The encoder is all zeros, so every
    # hidden state is exactly zero and the pooler dense reduces to its bias:
    #   pooled[c] = act(POOLER_BIAS) for every channel c,
    #   logits    = classifier.weight @ pooled  (classifier.bias = 0)
    # With the selector rows below the three logits are 1x, 2x and -1x that
    # single activation value, which is what Task 11 asserts against.
    t["pooler.dense.weight"] = r(H, H)
    t["pooler.dense.bias"] = np.full(H, POOLER_BIAS, dtype=np.float32)
    cw = np.zeros((NCLS, H), dtype=np.float32)
    cw[0, 0] = 1.0; cw[1, 1] = 2.0; cw[2, 2] = -1.0
    t["classifier.weight"] = cw
    t["classifier.bias"] = r(NCLS)
    save_file(t, str(dir_model / "model.safetensors"))
    return {"vocab_size": V, "spm_pieces": V_SPM, "mask_id": MASK_ID}

def main():
    with tempfile.TemporaryDirectory() as td:
        dm = pathlib.Path(td) / "m"; dm.mkdir()
        dims = build_synthetic(dm)
        out = pathlib.Path(td) / "m.gguf"
        subprocess.run([sys.executable, str(ROOT / "convert_hf_to_gguf.py"),
                        str(dm), "--outfile", str(out), "--outtype", "f32"], check=True)
        reader = gguf.GGUFReader(str(out))
        kv = {f.name: f for f in reader.fields.values()}
        def u32(n): return int(kv[n].parts[kv[n].data[0]])
        assert kv["general.architecture"].parts[kv["general.architecture"].data[0]].tobytes() == b"deberta"
        assert u32("deberta.attention.position_buckets") == 4
        assert u32("deberta.attention.max_relative_positions") == 16  # resolved from -1
        # Special-token contract. The UGM defaults (eos=1, unk=2, pad=0, bos NULL,
        # add_bos=false) are wrong at both ends for a DeBERTa vocab, so the
        # converter must write these explicitly. The synthetic spm is trained with
        # pad 0, unk 1, bos 2, eos 3.
        assert u32("tokenizer.ggml.bos_token_id") == 2
        assert u32("tokenizer.ggml.eos_token_id") == 3
        assert u32("tokenizer.ggml.padding_token_id") == 0
        assert u32("tokenizer.ggml.unknown_token_id") == 1
        assert u32("tokenizer.ggml.add_bos_token") == 1
        assert u32("tokenizer.ggml.add_eos_token") == 1
        # Token list is sized by config vocab_size, not the spm piece count, with
        # the tail filled by [PAD{i}]. The real checkpoint pads 128100 against a
        # smaller spm; without this the filler path would never be exercised.
        toks = kv["tokenizer.ggml.tokens"]
        assert len(toks.data) == dims["vocab_size"], (
            f'token list is {len(toks.data)} long, expected {dims["vocab_size"]} '
            f'(config vocab_size, not the {dims["spm_pieces"]} spm pieces)')
        last = toks.parts[toks.data[-1]].tobytes()
        assert last.startswith(b"[PAD"), f"tail token is {last!r}, expected a [PAD{{i}}] filler"
        # added_tokens.json must survive the filler loop. The real checkpoint's
        # [MASK] sits at the first filler id, and SpecialVocab writes
        # tokenizer.ggml.mask_token_id pointing at it regardless, so dropping
        # this block ships a mask id aimed at [PAD{i}] with no other symptom.
        mask_id = dims["mask_id"]
        got_mask = toks.parts[toks.data[mask_id]].tobytes()
        assert got_mask == b"[MASK]", (
            f"token {mask_id} is {got_mask!r}, expected b'[MASK]' from added_tokens.json; "
            f"set_vocab dropped conversion/base.py:1885-1897's added-tokens block")
        # CONTROL, not USER_DEFINED: added_tokens.json types it USER_DEFINED
        # (base.py:1897) and the tokenizer_config.json decoder block then
        # re-types it CONTROL because its entry is special: true
        # (base.py:1913-1914). Both attributes enter cache_special_tokens
        # (src/llama-vocab.cpp:2977-2979), so tokenisation is identical either
        # way and no NLI number moves; the difference is detokenisation, where
        # attr_special = UNKNOWN | CONTROL (:3561) suppresses a CONTROL token in
        # token_to_piece(special=false) and renders a USER_DEFINED one
        # literally. USER_DEFINED here means the decoder block was dropped.
        ttypes = kv["tokenizer.ggml.token_type"]
        got_type = int(ttypes.parts[ttypes.data[mask_id]])
        assert got_type == int(gguf.TokenType.CONTROL), (
            f"token {mask_id} has type {got_type}, expected CONTROL "
            f"({int(gguf.TokenType.CONTROL)}). USER_DEFINED "
            f"({int(gguf.TokenType.USER_DEFINED)}) means set_vocab ported "
            f"conversion/base.py:1885-1897 but not the decoder block at :1899-1920; "
            f"UNUSED means it dropped both and this is still the filler entry")
        names = {t.name for t in reader.tensors}
        required = {"rel_embd.weight", "rel_embd_norm.weight", "rel_embd_norm.bias",
                    "token_embd.weight", "token_embd_norm.weight", "token_embd_norm.bias",
                    "cls.weight", "cls.bias", "cls.output.weight", "cls.output.bias"}
        for i in range(2):
            for s in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_up", "ffn_down"):
                required.add(f"blk.{i}.{s}.weight"); required.add(f"blk.{i}.{s}.bias")
            required.add(f"blk.{i}.attn_output_norm.weight"); required.add(f"blk.{i}.attn_output_norm.bias")
            required.add(f"blk.{i}.layer_output_norm.weight"); required.add(f"blk.{i}.layer_output_norm.bias")
        missing = required - names
        assert not missing, f"converted GGUF is missing required tensors: {sorted(missing)}"
        # position_ids must have been dropped by filter_tensors, not mapped. The
        # real checkpoint ships it; without the skip the conversion raises
        # "Can not map tensor" long before this assertion is reached, so a green
        # run here proves both that the skip exists and that it names the right key.
        stray = {n for n in names if "position_ids" in n}
        assert not stray, f"position_ids leaked into the GGUF: {sorted(stray)}"
        print("test-deberta-convert: OK")

if __name__ == "__main__":
    main()
