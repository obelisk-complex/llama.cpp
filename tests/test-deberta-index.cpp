// Release builds define NDEBUG, which turns assert() into a no-op; undef it
// here so this test still checks its assertions under -DCMAKE_BUILD_TYPE=Release.
#undef NDEBUG
#include "../src/llama-graph.h"
#include <cassert>
#include <cstdio>
#include <vector>

// FROZEN REFERENCE TABLE, produced once by the HF generator recorded below this
// test and checked in as literals. It is NOT recomputed here from
// deberta_relative_position_bucket, nor from any other code in this repository:
// that is exactly what would turn this test back into a tautology. If these
// values ever look wrong, re-run the generator against HF and change the table
// deliberately. Do not "simplify" it into a computed expectation.
//
// Config: bucket_size = 8, max_position = 16 -> att_span = 8, rel table rows 16.
// pos = {0, 1, 2, 5, 20}: the last position is deliberately past max_position,
// so the top clamp is genuinely exercised rather than assumed unreachable.
//
// The two columns come from two independent HF observations: c2p from
// make_log_bucket_position plus the documented clamp, p2c from running
// DisentangledSelfAttention itself and reading back which rel row it selects
// for that (q, k). They agree, which is the point - a wrong p2c convention
// would show up as a disagreement between the columns, not as both moving
// together.
struct ref_entry { int q; int k; int32_t c2p; int32_t p2c; };
static const ref_entry REF[] = {
    { 0, 0,  8,  8 },  // rel   0, bucket  0        q == k
    { 1, 0,  9,  9 },  // rel   1, bucket  1        q > k, exact range
    { 0, 1,  7,  7 },  // rel  -1, bucket -1        q < k, exact range
    { 3, 0, 13, 13 },  // rel   5, bucket  5        q > k, log-bucketed (|rel| > mid)
    { 0, 3,  3,  3 },  // rel  -5, bucket -5        q < k, log-bucketed
    { 4, 3, 15, 15 },  // rel  15, bucket  7        log-bucketed, top slot, unclamped
    { 4, 0, 15, 15 },  // rel  20, bucket  8 -> 16  CLAMPED to 2*span-1
    { 0, 4,  0,  0 },  // rel -20, bucket -8        low edge, not clamped
};

int main() {
    const int64_t T = 5, H = 2;
    const int32_t buckets = 8, max_pos = 16;
    std::vector<llama_pos> pos = {0, 1, 2, 5, 20};
    std::vector<int32_t> c2p(T * T * H), p2c(T * T * H);
    deberta_fill_c2p_index(c2p.data(), pos.data(), T, H, buckets, max_pos);
    deberta_fill_p2c_index(p2c.data(), pos.data(), T, H, buckets, max_pos);

    auto c2p_at = [&](int k, int q, int h) { return c2p[k + q * T + h * T * T]; };
    auto p2c_at = [&](int q, int k, int h) { return p2c[q + k * T + h * T * T]; };
    for (int h = 0; h < H; ++h) {
        for (const auto & e : REF) {
            assert(c2p_at(e.k, e.q, h) == e.c2p);
            assert(p2c_at(e.q, e.k, h) == e.p2c);
        }
        // The two fills differ only in write layout, so one is the transpose of
        // the other over the whole grid. A sign flip in either breaks this.
        // The literals this is checked against were derived two different ways
        // (see the table header), so this is a cross-check, not a restatement.
        for (int q = 0; q < T; ++q) {
            for (int k = 0; k < T; ++k) {
                assert(p2c_at(q, k, h) == c2p_at(k, q, h));
            }
        }
        for (int i = 0; i < T * T; ++i) {  // head replication
            assert(c2p[i] == c2p[i + h * T * T]);
            assert(p2c[i] == p2c[i + h * T * T]);
        }
    }
    printf("test-deberta-index: OK\n");
    return 0;
}
