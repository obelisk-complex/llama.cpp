#include "models.h"

void llama_model_deberta::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,          hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_POSITION_BUCKETS,       hparams.position_buckets);
    ml.get_key(LLM_KV_ATTENTION_MAX_RELATIVE_POSITIONS, hparams.max_relative_positions);

    switch (hparams.n_layer()) {
        case 12: type = LLM_TYPE_109M; break; // deberta-v3-base
        case 24: type = LLM_TYPE_335M; break; // deberta-v3-large
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deberta::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_rel_rows = 2 * (int64_t) hparams.position_buckets;

    tok_embd   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab}, 0);
    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias",   0), {n_embd}, 0);

    rel_embd        = create_tensor(tn(LLM_TENSOR_REL_EMBD,      "weight"), {n_embd, n_rel_rows}, 0);
    rel_embd_norm   = create_tensor(tn(LLM_TENSOR_REL_EMBD_NORM, "weight"), {n_embd}, 0);
    rel_embd_norm_b = create_tensor(tn(LLM_TENSOR_REL_EMBD_NORM, "bias"),   {n_embd}, 0);

    // classification head (pooler dense + classifier). Optional so a bare encoder loads.
    cls       = create_tensor(tn(LLM_TENSOR_CLS,     "weight"), {n_embd, n_embd},            TENSOR_NOT_REQUIRED);
    cls_b     = create_tensor(tn(LLM_TENSOR_CLS,     "bias"),   {n_embd},                    TENSOR_NOT_REQUIRED);
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   {hparams.n_cls_out},         TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0); // separate q/k/v (+ biases)

        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd},         0);
        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias",   i), {n_embd}, 0);
        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias",   i), {n_ff},         0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd},       0);
        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias",   i), {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deberta::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

void llama_model_deberta::graph::pos_projections(const llama_model & model, int il, int64_t n_embd_head,
                                                 ggml_tensor ** pos_key, ggml_tensor ** pos_query) {
    const int64_t n_rel_rows = model.rel_embd->ne[1]; // 2*att_span
    // share_att_key: project rel through this layer's content Wk / Wq, biases included.
    ggml_tensor * pk = build_lora_mm(model.layers[il].wk, rel);
    if (model.layers[il].wk_b) pk = ggml_add(ctx0, pk, model.layers[il].wk_b);
    ggml_tensor * pq = build_lora_mm(model.layers[il].wq, rel);
    if (model.layers[il].wq_b) pq = ggml_add(ctx0, pq, model.layers[il].wq_b);
    pk = ggml_reshape_3d(ctx0, pk, n_embd_head, n_head, n_rel_rows);       // [d, H, 2S]
    pq = ggml_reshape_3d(ctx0, pq, n_embd_head, n_head, n_rel_rows);
    *pos_key   = ggml_cont(ctx0, ggml_permute(ctx0, pk, 0, 2, 1, 3));      // [d, 2S, H]
    *pos_query = ggml_cont(ctx0, ggml_permute(ctx0, pq, 0, 2, 1, 3));
}

llama_model_deberta::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // scale_factor = 1 + |pos_att_type| = 3. Applied to (content + disentangled)
    // sum inside soft_max_ext. Content-only baseline uses the same scale.
    const float kq_scale = 1.0f / sqrtf(3.0f * float(n_embd_head));

    ggml_tensor * inpL = build_inp_embd(model.tok_embd); // token only (no type, no abs pos)
    cb(inpL, "inp_embd", -1);
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, 0);
    cb(inpL, "inp_norm", 0);

    auto * inp_attn = build_attn_inp_no_cache();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // norm_rel_ebd == "layer_norm": once, shared across layers.
    rel = build_norm(model.rel_embd, model.rel_embd_norm, model.rel_embd_norm_b, LLM_NORM, -1);
    cb(rel, "rel_embd_norm", -1);
    auto * inp_pos = build_inp_deberta_pos();

    // Declared outside the loop because it is read again after it (`cur = inpL`
    // below), exactly as src/models/bert.cpp:85 declares its own.
    ggml_tensor * cur;

    for (int il = 0; il < n_layer; ++il) {
        cur = inpL;
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);
            cb(Qcur, "Qcur", il); cb(Kcur, "Kcur", il); cb(Vcur, "Vcur", il);

            // per-head content Q, K as [d, n_tokens, H]
            ggml_tensor * Qh = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 0, 2, 1, 3));
            ggml_tensor * Kh = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            ggml_tensor * pos_key = nullptr, * pos_query = nullptr;
            pos_projections(model, il, n_embd_head, &pos_key, &pos_query);

            ggml_tensor * kq_b = ggml_add(ctx0,
                    deberta_c2p_bias(ctx0, pos_key,   Qh, inp_pos->c2p_index),
                    deberta_p2c_bias(ctx0, pos_query, Kh, inp_pos->p2c_index)); // [n_kv, n_q, H], unscaled
            cb(kq_b, "disentangled_bias", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, nullptr,
                    Qcur, Kcur, Vcur, kq_b, nullptr, nullptr, kq_scale, il);
            cb(cur, "kqv_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = ggml_add(ctx0, cur, inpL);
        cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL,
                LLM_FFN_GELU_ERF, LLM_FFN_SEQ, il); // HF "gelu" is exact erf
        cb(cur, "ffn_out", il);
        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);
        inpL = cur;
    }

    cur = inpL;
    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    // The framework applies build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm)
    // after build_arch_graph (src/llama-model.cpp:2418), reading model.cls* directly;
    // the graph must NOT call it here (that would double-pool). Task 11 adds the
    // DeBERTa activation branch inside build_pooling itself.

    ggml_build_forward_expand(gf, cur);
}
