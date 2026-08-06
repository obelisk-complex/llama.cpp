// Release builds define NDEBUG, which turns assert() into a no-op; undef it
// here so this test still checks its assertions under -DCMAKE_BUILD_TYPE=Release.
#undef NDEBUG
#include "../src/llama-graph.h"
#include <cassert>
#include <cmath>
#include <cstdio>

// Reference from HF make_log_bucket_position(rel, 256, 512): mid = 128.
//   |rel| <= mid -> bucket = rel; else sign*(ceil(log(|rel|/mid)/log(511/128)*127)+128).
static int32_t ref_large(int32_t rel) {
    const int32_t sign = (rel > 0) - (rel < 0);
    const int32_t mid  = 128;
    const double  a    = (double) std::abs(rel);
    const int32_t lp   = (int32_t) std::ceil(std::log(a / mid) / std::log(511.0 / mid) * (mid - 1)) + mid;
    return sign * lp;
}

int main() {
    assert(deberta_relative_position_bucket(0,    256, 512) == 0);
    assert(deberta_relative_position_bucket(1,    256, 512) == 1);
    assert(deberta_relative_position_bucket(-1,   256, 512) == -1);
    assert(deberta_relative_position_bucket(127,  256, 512) == 127);
    assert(deberta_relative_position_bucket(-128, 256, 512) == -128);
    assert(deberta_relative_position_bucket(128,  256, 512) == 128);
    assert(deberta_relative_position_bucket(200,  256, 512) == ref_large(200));
    assert(deberta_relative_position_bucket(-200, 256, 512) == ref_large(-200));
    assert(deberta_relative_position_bucket(511,  256, 512) == ref_large(511));
    assert(deberta_relative_position_bucket(-511, 256, 512) == ref_large(-511));
    for (int32_t x = 1; x < 512; ++x) {
        assert(deberta_relative_position_bucket(x, 256, 512) ==
               -deberta_relative_position_bucket(-x, 256, 512));
    }
    printf("test-deberta-bucket: OK\n");
    return 0;
}
