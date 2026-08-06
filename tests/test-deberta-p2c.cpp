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
    const int64_t d = 2, H = 1, T = 2, S2 = 4;
    struct ggml_init_params p = { 16*1024*1024, NULL, false };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * pos_query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, S2, H);
    float pq[8] = { 1,0, 2,0, 3,0, 4,0 };   // col b: [b+1, 0]
    memcpy(pos_query->data, pq, sizeof(pq));

    ggml_tensor * Kh = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, T, H);
    float k[4] = { 1,0, 0,1 };               // K_0=(1,0), K_1=(0,1)
    memcpy(Kh->data, k, sizeof(k));

    // p2c_index[q + k*T]: [q0k0]=1,[q1k0]=2,[q0k1]=3,[q1k1]=0
    ggml_tensor * idx = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, T, T, H);
    int32_t ix[4] = { 1, 2, 3, 0 };
    memcpy(idx->data, ix, sizeof(ix));

    ggml_tensor * out = deberta_p2c_bias(ctx, pos_query, Kh, idx); // [n_kv, n_q, H]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // pre-transpose result_pre[q + k*T] = K_k . pos_query[idx[q,k]]:
    //  [q0k0]=K0.pq1=2,[q1k0]=K0.pq2=3,[q0k1]=K1.pq3=0,[q1k1]=K1.pq0=0
    // after transpose out[k + q*T] = result_pre[q,k]:
    //  [k0q0]=2,[k1q0]=0,[k0q1]=3,[k1q1]=0
    const float * o = (const float *) out->data;
    const float exp_[4] = { 2, 0, 3, 0 };
    for (int i = 0; i < 4; ++i) assert(std::fabs(o[i] - exp_[i]) < 1e-5f);

    printf("test-deberta-p2c: OK\n");
    ggml_free(ctx);
    return 0;
}
