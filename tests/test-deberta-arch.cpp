// Release builds define NDEBUG, which turns assert() into a no-op; undef it
// here so this test still checks its assertions under -DCMAKE_BUILD_TYPE=Release.
#undef NDEBUG
#include "../src/llama-arch.h"
#include <cassert>
#include <cstdio>
#include <string>

int main() {
    assert(llm_arch_from_string("deberta") == LLM_ARCH_DEBERTA);
    assert(std::string(llm_arch_name(LLM_ARCH_DEBERTA)) == "deberta");

    LLM_KV kv(LLM_ARCH_DEBERTA);
    assert(kv(LLM_KV_ATTENTION_POSITION_BUCKETS)       == "deberta.attention.position_buckets");
    assert(kv(LLM_KV_ATTENTION_MAX_RELATIVE_POSITIONS) == "deberta.attention.max_relative_positions");

    // Assert through the public accessor, not the table. LLM_TENSOR_NAMES is a
    // file-static in llama-arch.cpp with no header declaration, so naming it here
    // does not compile; and LLM_TN_IMPL::str() is the only path production code
    // (Task 6's loader) ever takes from a llm_tensor to a string. Testing the
    // table would prove the table and leave the accessor unproven.
    LLM_TN tn(LLM_ARCH_DEBERTA);
    assert(std::string("rel_embd.weight")      == tn(LLM_TENSOR_REL_EMBD,      "weight"));
    assert(std::string("rel_embd_norm.weight") == tn(LLM_TENSOR_REL_EMBD_NORM, "weight"));

    printf("test-deberta-arch: OK\n");
    return 0;
}
