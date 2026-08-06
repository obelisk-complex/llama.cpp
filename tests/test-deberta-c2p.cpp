// Release builds define NDEBUG, which turns assert() into a no-op; undef it
// here so this test still checks its assertions under -DCMAKE_BUILD_TYPE=Release.
#undef NDEBUG
#include "../src/llama-graph.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

int main() {
    // d=2, H=1, T=2, att_span S=2 -> 2S=4.
    const int64_t d = 2, H = 1, T = 2, S2 = 4;
    struct ggml_init_params p = { 16*1024*1024, NULL, false };
    ggml_context * ctx = ggml_init(p);

    // pos_key[:,b,0] = (b+1, 0)  -> Q_q . pos_key_b uses only the first component
    ggml_tensor * pos_key = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, S2, H);
    float pk[8] = { 1,0, 2,0, 3,0, 4,0 }; // (col b: [b+1, 0])
    memcpy(pos_key->data, pk, sizeof(pk));

    // Qh[:,q,0]: Q_0=(1,0), Q_1=(0,1)
    ggml_tensor * Qh = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, T, H);
    float q[4] = { 1,0, 0,1 };
    memcpy(Qh->data, q, sizeof(q));

    // c2p_index[k + q*T]: [k0q0]=2,[k1q0]=0,[k0q1]=3,[k1q1]=1
    ggml_tensor * idx = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, T, T, H);
    int32_t ix[4] = { 2, 0, 3, 1 };
    memcpy(idx->data, ix, sizeof(ix));

    ggml_tensor * out = deberta_c2p_bias(ctx, pos_key, Qh, idx); // [n_kv, n_q, H]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // expected out[k + q*T]: Q_q . pos_key[idx]:
    //  [k0q0]=Q0.pk2=3, [k1q0]=Q0.pk0=1, [k0q1]=Q1.pk3=0, [k1q1]=Q1.pk1=0
    const float * o = (const float *) out->data;
    const float exp_[4] = { 3, 1, 0, 0 };
    for (int i = 0; i < 4; ++i) assert(std::fabs(o[i] - exp_[i]) < 1e-5f);

    printf("test-deberta-c2p: OK\n");
    ggml_free(ctx);
    return 0;
}
