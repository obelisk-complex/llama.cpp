from __future__ import annotations
from typing import Callable, Iterable
from torch import Tensor
import json
from .base import ModelBase, TextModel, gguf, logger


# Only the architecture this plan proves end to end. The HF architecture names
# "DebertaV2Model" and "DebertaV2ForTokenClassification" (distinct from the
# conversion class below, which merely shares a spelling with the first) are
# deliberately NOT registered: the
# unconditional add_pooling_type(RANK) and the pooler.dense -> cls /
# classifier -> cls.output renames below are correct for the sequence
# classifier alone. A token classifier has no ContextPooler, but its
# `classifier` still maps to cls.output, its id2label still sets a plausible
# n_cls_out, and build_pooling's RANK branch guards its pooler on `if (cls)`
# (src/llama-graph.cpp:3631), so such a model would convert cleanly, build a
# valid graph, and return exactly num_labels floats computed from token 0
# alone - right for the first token, silently wrong for every other one, with
# no error at any layer. Adding either name needs pooling selected per
# architecture and the classifier rename skipped for the token classifier,
# plus its own fidelity fixture; that is a later, separate stage.
#
# Registering "DebertaV2Model" purely to raise a friendlier error was
# considered and rejected: it would never fire for the models that would
# benefit. microsoft/deberta-v3-{xsmall,small,base,large} and
# mdeberta-v3-base ship a config.json with no "architectures" key at all,
# so get_model_architecture leaves arch None and raises "Failed to detect
# model architecture" (conversion/base.py:2645) before any register lookup
# happens. A registration keyed on the name is only reachable from a
# checkpoint that declares that name, which the canonical ones do not.
# The supported shape is documented in the fork README instead (Task 11a).
@ModelBase.register("DebertaV2ForSequenceClassification")
class DebertaV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DEBERTA

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_layer_norm_eps(self.hparams.get("layer_norm_eps", 1e-7))
        self.gguf_writer.add_pooling_type(gguf.PoolingType.RANK)

        # DeBERTa-v2 proper carries an encoder ConvLayer this port implements
        # nowhere. transformers gates it on this field alone, and all three
        # released v2 models set conv_kernel_size 3 while no v3 model sets it
        # at all. Refuse by name here: without this the conversion instead dies
        # on "Can not map tensor 'deberta.encoder.conv.conv.weight'", which
        # reads as a mapping gap rather than an unimplemented feature.
        if int(self.hparams.get("conv_kernel_size", 0)) > 0:
            raise ValueError(
                "this DeBERTa port does not implement the DeBERTa-v2 encoder ConvLayer "
                f"(conv_kernel_size={self.hparams.get('conv_kernel_size')}, "
                f"conv_act={self.hparams.get('conv_act')!r}). Every released DeBERTa-v2 "
                "checkpoint carries it; the DeBERTa-v3 line does not.")

        if self.hparams.get("position_biased_input", False):
            raise ValueError("position_biased_input=True is unsupported by this DeBERTa port")

        # The graph hardcodes each of the following; only position_biased_input
        # was refused before. All eight real configs checked are uniform and
        # match every hardcode, so the exposure is fine-tunes and future
        # variants, where a changed field converts silently and returns wrong
        # logits with nothing to point at. Each guard names its field.
        #
        # pos_att_type fixes both scale_factor = 3 (Task 6's kq_scale) and the
        # presence of both bias terms (Task 10). DebertaV2Config normalises it
        # with split("|"), and the real configs use both spellings: the string
        # "p2c|c2p" in every microsoft config, a list only in the cross-encoder
        # fine-tune this plan validates against. Match the normalisation, or a
        # guard written against the list form alone crashes on
        # microsoft/deberta-v3-base.
        pos_att_type = self.hparams.get("pos_att_type") or []
        if isinstance(pos_att_type, str):
            pos_att_type = pos_att_type.split("|")
        if sorted(str(x).strip().lower() for x in pos_att_type) != ["c2p", "p2c"]:
            raise ValueError(
                f"this DeBERTa port hardcodes pos_att_type = p2c + c2p (scale_factor 3); "
                f"got {self.hparams.get('pos_att_type')!r}")
        if not self.hparams.get("relative_attention", False):
            raise ValueError("relative_attention=False is unsupported by this DeBERTa port")
        if not self.hparams.get("share_att_key", False):
            raise ValueError(
                "share_att_key=False is unsupported: the graph projects the relative "
                "embeddings through each layer's own Wq/Wk")
        if self.hparams.get("norm_rel_ebd") != "layer_norm":
            raise ValueError(
                f"this DeBERTa port hardcodes norm_rel_ebd=layer_norm; "
                f"got {self.hparams.get('norm_rel_ebd')!r}")
        if int(self.hparams.get("type_vocab_size", 0)) != 0:
            raise ValueError(
                f"segment embeddings are unsupported by this DeBERTa port "
                f"(type_vocab_size={self.hparams.get('type_vocab_size')})")
        if self.hparams.get("pooler_hidden_act", "gelu") != "gelu":
            raise ValueError(
                f"build_pooling's DeBERTa branch hardcodes the erf GELU ContextPooler "
                f"activation; got pooler_hidden_act={self.hparams.get('pooler_hidden_act')!r}")

        id2label = self.hparams.get("id2label")
        if id2label:  # sets n_cls_out on the C++ side (label count)
            # config.json gives the label indices as strings, so a plain sort is
            # lexicographic: correct for {"0","1","2"}, but "0","1","10","11","2"
            # at ten or more labels. n_cls_out takes the length and is unaffected,
            # so only the GGUF label strings would misorder - latent against the
            # general-family claim rather than broken for this model.
            self.gguf_writer.add_classifier_output_labels(
                [v for _, v in sorted(id2label.items(), key=lambda kv: int(kv[0]))])

        position_buckets = int(self.hparams.get("position_buckets", -1))
        max_rel = int(self.hparams.get("max_relative_positions", -1))
        if max_rel < 1:
            max_rel = int(self.hparams["max_position_embeddings"])  # resolve -1
        if position_buckets <= 0:
            position_buckets = max_rel  # att_span == max_rel when no buckets
        self.gguf_writer.add_position_buckets(position_buckets)
        self.gguf_writer.add_max_relative_positions(max_rel)

    def set_vocab(self):
        # Vocab SHAPE only: the piece list, its scores and its types, sized by
        # the config's vocab_size. Task 5a adds the contents - the two
        # added-token blocks from conversion/base.py, the spm normaliser
        # settings, and the explicit special-token ids - to this same method.
        # DeBERTa-v3 ships a SentencePiece unigram model as spm.model (the base
        # helper reads tokenizer.model), so load spm.model explicitly.
        from sentencepiece import SentencePieceProcessor
        spm_path = self.dir_model / "spm.model"
        if not spm_path.is_file():
            raise FileNotFoundError(f"DeBERTa-v3 expects spm.model at {spm_path}")
        sp = SentencePieceProcessor(); sp.LoadFromFile(str(spm_path))
        vocab_size = int(self.hparams.get("vocab_size", sp.vocab_size()))

        tokens = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores = [-10000.0] * vocab_size
        toktypes = [gguf.TokenType.UNUSED] * vocab_size
        for tid in range(min(sp.vocab_size(), vocab_size)):
            tokens[tid] = sp.IdToPiece(tid).encode("utf-8")
            scores[tid] = sp.GetScore(tid)
            if sp.IsUnknown(tid):   toktypes[tid] = gguf.TokenType.UNKNOWN
            elif sp.IsControl(tid): toktypes[tid] = gguf.TokenType.CONTROL
            elif sp.IsUnused(tid):  toktypes[tid] = gguf.TokenType.UNUSED
            else:                   toktypes[tid] = gguf.TokenType.NORMAL

        self.gguf_writer.add_tokenizer_model("t5")   # spm unigram
        self.gguf_writer.add_tokenizer_pre("default")

        # Added tokens, ported from conversion/base.py:1885-1897 (the block the
        # tokenizer.model helper carries and Task 5's hand-written loop would
        # otherwise drop, guard included). Line-for-line the same logic; the one
        # difference is that base.py names the enum SentencePieceTokenTypes
        # (base.py:61) where this uses gguf.TokenType (constants.py:4932-4938).
        # The two enums carry identical values and newer base.py code already
        # mixes them, so this is a naming choice, not a dropped conversion.
        # DeBERTa-v3's added_tokens.json is {"[MASK]": 128000}, and its
        # 128000-piece spm against a config vocab_size of 128100 puts 128000
        # squarely in the filler tail: without this the token ships as
        # [PAD128000]/UNUSED/-10000 while SpecialVocab still writes
        # tokenizer.ggml.mask_token_id = 128000 ('mask' is in the default
        # special_token_types at gguf-py/gguf/vocab.py:74, and add_mask_token_id
        # exists at gguf_writer.py:1130), so the shipped mask id points at a
        # filler token. [PAD128000] is the name Task 5's own filler loop gives
        # id 128000, keyed on the token id; base.py's separate filler loop
        # (:1922-1926) numbers by count instead, [PAD1] through [PAD{pad_count}].
        # Both are correct; do not "fix" Task 5's to match base.py's, because
        # Task 5's tail assertion is written against the id-keyed names.
        # Inert for NLI, which never emits [MASK]; not inert for the DeBERTa-v3
        # line this fork is published for.
        added_tokens_file = self.dir_model / "added_tokens.json"
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
            for key in added_tokens_json:
                token_id = added_tokens_json[key]
                if token_id >= vocab_size:
                    logger.warning(f"ignore token {token_id}: id is out of range, max={vocab_size - 1}")
                    continue
                tokens[token_id] = key.encode("utf-8")
                scores[token_id] = -1000.0
                toktypes[token_id] = gguf.TokenType.USER_DEFINED

        # base.py's adjacent tokenizer_config.json / added_tokens_decoder block
        # (:1899-1920), ported too. It is NOT redundant with Task 5's spm loop:
        # the validation checkpoint's own tokenizer_config.json carries
        # added_tokens_decoder = {0:[PAD], 1:[CLS], 2:[SEP], 3:[UNK],
        # 128000:[MASK]}, every entry special:true, and 128000 is past the
        # 128000-piece spm - so the decoder names a token the spm does not
        # carry, and it is [MASK], the same token the added-tokens block above
        # types USER_DEFINED. base.py:1913-1914 types it CONTROL instead.
        # does_token_look_special is already on ModelBase (:1318).
        tokenizer_config_file = self.dir_model / "tokenizer_config.json"
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
            added_tokens_decoder = tokenizer_config_json.get("added_tokens_decoder", {})
            for token_id, token_data in added_tokens_decoder.items():
                token_id = int(token_id)
                token: str = token_data["content"]
                if token_id >= vocab_size:
                    logger.warning(f"ignore token {token_id}: id is out of range, max={vocab_size - 1}")
                    continue
                # base.py:1910-1912's warning, ported with the rest. base.py
                # writes it as a nested if; the `and` below is the same
                # condition, flattened. Inert on the validation checkpoint (ids
                # 0-3 carry the same strings the spm does), but it is the only
                # diagnostic that fires when a fine-tune's added_tokens_decoder
                # disagrees with its own spm vocabulary - the general DeBERTa-v3
                # case this port claims to support - and the overwrite two lines
                # below is silent without it.
                if toktypes[token_id] != gguf.TokenType.UNUSED and tokens[token_id] != token.encode("utf-8"):
                    logger.warning(f"replacing token {token_id}: {tokens[token_id].decode('utf-8')!r} -> {token!r}")
                if token_data.get("special") or self.does_token_look_special(token):
                    toktypes[token_id] = gguf.TokenType.CONTROL
                else:
                    token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")
                    toktypes[token_id] = gguf.TokenType.USER_DEFINED
                scores[token_id] = -1000.0
                tokens[token_id] = token.encode("utf-8")

        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        # Normaliser settings from spm.model's own normalizer_spec, as
        # conversion/t5.py:53-55, 112-115 does. precompiled_charsmap is optional
        # on the C++ side (src/llama-vocab.cpp:2047-2048), so omitting it drops
        # the spm normaliser silently rather than failing loudly.
        from sentencepiece import sentencepiece_model_pb2 as spm_pb2
        proto = spm_pb2.ModelProto()
        proto.ParseFromString(spm_path.read_bytes())
        norm = proto.normalizer_spec
        self.gguf_writer.add_add_space_prefix(norm.add_dummy_prefix)
        self.gguf_writer.add_remove_extra_whitespaces(norm.remove_extra_whitespaces)
        if norm.precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(norm.precompiled_charsmap)

        # SpecialVocab first, explicit ids second: duplicate keys warn and the
        # last write wins (gguf-py/gguf/gguf_writer.py:277-281).
        special = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special.add_to_gguf(self.gguf_writer)

        # SpecialVocab alone does NOT establish the special-token contract here.
        # tokenizer_model "t5" selects LLAMA_VOCAB_TYPE_UGM, whose defaults are
        # eos=1, unk=2, pad=0 with bos/sep/mask NULL (src/llama-vocab.cpp:2036-2045)
        # and add_bos=false, add_eos=true (:2412-2415). DeBERTa-v3 is [PAD]=0,
        # [CLS]=1, [SEP]=2, [UNK]=3, and HF emits "[CLS] ... [SEP]", so leaving
        # the defaults gives no leading [CLS] and a trailing [CLS]: wrong at both
        # ends. DeBERTa's tokenizer_config.json names cls_token/sep_token and no
        # bos/eos, so SpecialVocab writes neither.
        #
        # BOS is the carrier for [CLS]: LLM_KV_TOKENIZER_CLS_ID exists
        # (src/llama-arch.cpp:344) but llama-vocab.cpp never reads it, and gguf-py
        # has no add_cls_token_id, so SpecialVocab only warns and skips 'cls'
        # (gguf-py/gguf/vocab.py:89-92).
        def piece_id(piece: str, default: int) -> int:
            tid = sp.PieceToId(piece)
            return tid if tid >= 0 and sp.IdToPiece(tid) == piece else default

        # The defaults are the ids trained into the synthetic fixture's spm
        # (pad 0, unk 1, bos 2, eos 3); a real DeBERTa-v3 resolves all four by name.
        self.gguf_writer.add_bos_token_id(piece_id("[CLS]", 2))
        self.gguf_writer.add_eos_token_id(piece_id("[SEP]", 3))
        self.gguf_writer.add_sep_token_id(piece_id("[SEP]", 3))
        self.gguf_writer.add_unk_token_id(piece_id("[UNK]", 1))
        self.gguf_writer.add_pad_token_id(piece_id("[PAD]", 0))
        self.gguf_writer.add_add_bos_token(True)   # HF prepends [CLS]
        self.gguf_writer.add_add_eos_token(True)   # HF appends [SEP]

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # A persistent I64 buffer, not a weight: the validation checkpoint ships
        # it and nothing maps it, so map_tensor_name would raise.
        # conversion/bert.py:82 drops the same buffer, keyed unprefixed because
        # it strips "bert." first; this class does no prefix stripping, so match
        # on the suffix rather than the whole key. An exact match on
        # "deberta.embeddings.position_ids" would miss a checkpoint saved from a
        # bare DebertaV2Model, which carries the buffer with no prefix at all.
        #
        # The buffer is a property of the 2021-era transformers that saved this
        # particular checkpoint, not of the family: deberta-v2-xlarge's state
        # dict contains no position_ids whatsoever. So this skip must not be
        # read as something every DeBERTa checkpoint needs.
        if name.endswith("embeddings.position_ids"):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # pooler -> cls (rank pooling dense), classifier -> cls.output.
        if name.startswith("pooler.dense"):
            name = name.replace("pooler.dense", "cls")
        elif name.startswith("classifier"):
            name = name.replace("classifier", "cls.output")
        return [(self.map_tensor_name(name), data_torch)]
