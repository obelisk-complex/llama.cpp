# llama.cpp

> [!NOTE]
> **This is a fork** (`obelisk-complex/llama.cpp`, branch `wikiq-patches-b10288`) maintained for the
> wikiq project. Branched from release `b10046`, rebased
> 2026-08-05 onto `b10288` (242 upstream commits of drift; one inert enum-slot renumber was the only
> conflict). It carries:
>
> - Upstream PR [ggml-org/llama.cpp#21729](https://github.com/ggml-org/llama.cpp/pull/21729) (squashed): adds `token_type_ids` input for rerank models with type-embedding.
> - Upstream PR [ggml-org/llama.cpp#25448](https://github.com/ggml-org/llama.cpp/pull/25448) (cherry-picked): causal-LM reranker support via logit-margin scoring.
> - Four local rerank-fidelity fixes for `jina-bert-v2` (jinaai/jina-reranker-v1-turbo-en and siblings).
> - A fifth local fix for a crash on `bge-reranker-v2-m3` and other single-token-type BERT-arch rerankers, found while diagnosing what wikiq's own README called "order corruption" and turned out to be worse.
> - A sixth local fix: #21729's `token_type` decode divided by zero on an empty vocab and skipped the null-batch guard its sibling validation loop has, crashing (SIGFPE / SIGSEGV) rather than rejecting or defaulting cleanly. Found by the rebase's own test suite, not by rerank use - unrelated to the four fidelity fixes above.
> - A DeBERTa-v3 port (`LLM_ARCH_DEBERTA`): disentangled attention, the DeBERTa `ContextPooler`
>   classification head, and a `conversion/deberta.py` converter. See below.
>
> **Why:** wikiq uses jina-reranker-v1-turbo-en as its rerank stage, the step that takes a first-pass
> retrieval and puts the actually-relevant documents at the top before they're shown to a user or fed
> to an LLM. On stock llama.cpp + the official `ggml-org/jina-reranker-v1-turbo-en-GGUF` conversion,
> this stage was silently wrong: no crash, no error, no NaN; it just returned a confidently
> plausible-looking ranking that didn't match what the actual model would say. Raw scores were
> compressed into a narrow band (real range ~0.04-0.20, llama.cpp was returning ~0.06-0.08 for
> everything) and the resulting order was scrambled hard enough that the top real match (per the real
> model) could land in 3rd or 4th place. A wikiq CI gate that diffs live rerank order against genuine
> HF reference scores (rather than trusting llama.cpp's output on faith) is what caught this; nothing
> upstream flagged it, and a search of ggml-org/llama.cpp's issues/PRs at the time found no existing
> report of any of the four causes below.
>
> **What was actually wrong**, most-to-least impactful:
>   1. **Classification head layout.** The official GGUF conversion omits the `pooler.dense` layer HF's
>      forward pass actually applies before the classifier; llama.cpp was running `tanh(classifier(CLS))`
>      instead of the real `classifier(tanh(pooler_dense(CLS)))`. This alone accounts for most of the
>      score compression. Loader (`src/models/jina-bert-v2.cpp`) now supports both the single-tensor
>      direct-projection layout and this pooler-style two-tensor layout; the GGUF itself still needs
>      rebuilding to supply the missing tensors: see
>      [`scripts/wikiq/fix-rerank-gguf.py`](scripts/wikiq/fix-rerank-gguf.py).
>   2. **Tokeniser:** `jina-v1-en` was sharing `GPT2`'s byte-level pre-tokeniser with unrelated
>      tokeniser families, silently dropping every uppercase codepoint ("Which" tokenised as if it read
>      "hich") and mis-splitting mixed alphanumerics ("v2", "m3"). Any query or document with capitals
>      or version-like tokens was scored against a corrupted input. New dedicated
>      `LLAMA_VOCAB_PRE_TYPE_JINA_V1_EN` (`src/llama-vocab.{h,cpp}`) with the model's real word-level,
>      lowercase-normalised tokenisation.
>   3. **ALiBi head slopes:** jina-bert-v2's reference implementation halves the interpolated ALiBi
>      slope for 4 of its 12 attention heads (a documented "quick fix" in its own `modeling_bert.py`);
>      ggml's slope table has no way to express that, so those 4 heads applied roughly double the
>      intended distance penalty on every layer, on every token, compounding across the whole forward
>      pass. Loader now supports head-padding to 16 heads so the fixed GGUF (above) can permute real
>      heads onto ggml's matching standard slopes.
>   4. **GELU approximation.** HF's `JinaBertGLUMLP` uses the exact erf-form GELU, not the tanh
>      approximation `build_ffn` had wired up for this arch, a smaller (~0.003) but real contribution
>      to the mismatch. New `LLM_FFN_GELU_ERF`/`LLM_FFN_GEGLU_ERF` ops (`src/llama-graph.{h,cpp}`), used
>      only by `LLM_ARCH_JINA_BERT_V2`.
>
> **What this fixes in practice:** jina-reranker-v1-turbo-en can now actually be trusted as a rerank
> stage on this fork, verified against the real HF forward pass rather than assumed correct because it
> runs without error. Live scores agree with the real HF reference within ±0.005 (raw logit) and rank
> order matches exactly on wikiq's conformance fixture, including a near-tied pair the pre-fix build
> got wrong.
>
> **A fifth, separate fix: `bge-reranker-v2-m3` crashed on every rerank request.** PR #21729's document
> token-type marking assumes any `LLM_ARCH_BERT` model has a second token-type row; `bge-reranker-v2-m3`
> (XLM-RoBERTa-based, `type_vocab_size = 1`) does not, so the marking indexed past the end of a
> single-row embedding table: `GGML_ASSERT` abort on CPU, silent out-of-bounds reads on backends
> without that check. This is the actual failure mode behind wikiq's own "order corruption on some
> backends" description; a crash, not a subtler reordering. Fixed by gating the marking on
> `vocab->n_token_types() > 1` (`tools/server/server-common.cpp`) so true two-segment BERT rerankers
> keep #21729's original behaviour and single-type-vocab models match their own HF reference (which
> also emits all-zero token-type ids for this architecture). A separate, smaller data issue remains in
> the model's public GGUF conversion (missing `tokenizer.ggml.add_sep_token`, same class of bug as the
> jina fixes above) - see
> [`scripts/wikiq/fix-bge-rerank-gguf.py`](scripts/wikiq/fix-bge-rerank-gguf.py).
> `bge-reranker-v2-m3` is not wikiq's pinned reranker; this is a fork correctness fix
> found along the way, not evidence wikiq uses this model.
>
> **DeBERTa-v3 support (`LLM_ARCH_DEBERTA`).** This is the reason to pin the fork rather than build
> upstream: stock llama.cpp has no DeBERTa architecture, and a DeBERTa checkpoint stops at
> `Model DebertaV2ForSequenceClassification is not supported`. The fork adds the architecture, its
> position-bucket GGUF metadata keys and relative-embedding tensors, DeBERTa's own log-bucket
> relative-position arithmetic (`make_log_bucket_position`, `ceil`-based and signed, distinct from
> T5's `floor`-based bucketing and not interchangeable with it), a `conversion/deberta.py` converter
> class, the encoder graph with the three-term disentangled attention (content-to-content,
> content-to-position, position-to-content) composed from existing ggml ops rather than a new one,
> and the DeBERTa `ContextPooler` GELU branch in `build_pooling` so the classification head matches
> HF instead of applying `tanh`.
>
> **What it supports, and what it refuses.** Sequence classifiers only: the converter registers
> `DebertaV2ForSequenceClassification` and nothing else. That excludes more than it sounds like,
> because `microsoft/deberta-v3-{xsmall,small,base,large}` and `mdeberta-v3-base` ship a
> `config.json` with no `architectures` key at all. They are base encoders with no classification
> head to pool, they are not supported in v1, and converting one fails with the generic
> `Failed to detect model architecture`, which names neither DeBERTa nor this fork. Point the
> converter at a fine-tuned sequence classifier instead; `cross-encoder/nli-deberta-v3-base` is
> the checkpoint this port is proven against. Also refused, each named at conversion time rather
> than left to surface later as an unmappable tensor: `conv_kernel_size > 0`, the DeBERTa-v2
> encoder ConvLayer, which this port implements nowhere (every released DeBERTa-v2 checkpoint sets
> it and no v3 checkpoint does, which is why v3 is the supported line); `position_biased_input:
> true`; `relative_attention: false`; `share_att_key: false`, since the graph projects the
> relative embeddings through each layer's own `Wq`/`Wk`; any `norm_rel_ebd` other than
> `layer_norm`; any `type_vocab_size` other than 0, so no segment embeddings; any `pos_att_type`
> other than p2c plus c2p; and any `pooler_hidden_act` other than `gelu`. CPU-only and f32 in v1:
> no quantised path and no Vulkan or ROCm shader work.
>
> **Converting and serving.**
> ```console
> python3 convert_hf_to_gguf.py /path/to/nli-deberta-v3-base --outtype f32 \
>     --outfile /path/to/nli-deberta-v3-base-F32.gguf
> llama-server --embeddings --pooling rank -m /path/to/nli-deberta-v3-base-F32.gguf
> ```
> Score through `/v1/embeddings`, and **set `"embd_normalize": -1` in the request body**. The server
> euclidean-normalises any pooled output by default (`tools/server/server-task.h`), which rescales
> the classifier logits into a unit vector: no error, no warning, just plausible numbers that are
> not the model's. `llama-embedding` has the same default and the same remedy,
> `--embd-normalize -1`. Under `--pooling rank` the response carries exactly `n_cls_out` values,
> three for a 3-way NLI head.
>
> **Tests.** Eight `test-deberta-*` ctest targets cover the architecture registration, the bucket
> arithmetic, the index invariant, the c2p and p2c bias terms, the conversion round-trip, the
> encoder, and the classification head:
> ```console
> cmake -B build -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON
> cmake --build build -j"$(nproc)"
> ctest --test-dir build -R '^test-deberta' --no-tests=error --output-on-failure
> ```
> That must report 8 tests; fewer means cmake found no Python interpreter with the converter's
> requirements and three of the eight were never registered. Those three convert a checkpoint, so
> they need `requirements/requirements-convert_hf_to_gguf.txt` installed in the interpreter cmake
> resolved at configure time.
>
> Stock llama.cpp has nothing to compare against here: the same checkpoint stops at
> `Model DebertaV2ForSequenceClassification is not supported` before a GGUF exists. The
> before-and-after worth running is the rerank one below, where both builds produce numbers.
>
> **A separate fix, not DeBERTa-specific: RANK-pooled embedding reads over-ran the heap.** Under
> `--pooling rank`, `llama_context` sizes each sequence's embedding buffer at `n_cls_out` floats
> (`src/llama-context.cpp`), but `llama-embedding` and `llama-server` both read `n_embd_out` of
> them: on a 3-class model that is 3072 bytes read out of a 12-byte allocation. The tail was not
> padding. `common_embd_normalize` writes every output it is given, so the zero-init of the
> response vector was overwritten with out-of-bounds reads and then serialised into the HTTP
> response; under the default euclidean norm the divisor summed over those reads as well, so the
> real logits were corrupted too, not just the values after them. Both call sites now clamp the
> read to `n_cls_out`, and `/v1/embeddings` on a RANK-pooled model returns that many values and
> nothing after them. This bug is upstream's, present at `b10288`, and it reaches any RANK-pooled
> model on any architecture; it is carried as its own commit for that reason. `/v1/rerank` is
> served by `send_rerank`, which reads `embd[0]` alone, and is unaffected.
>
> **Checking the fixes yourself.** Every claim above is a before-and-after you can run. Build stock
> and this fork from the same base, convert the same model, and compare:
> ```console
> git clone https://github.com/obelisk-complex/llama.cpp && cd llama.cpp
> git checkout b10288 && cmake -B build-stock && cmake --build build-stock -j"$(nproc)"
> git checkout wikiq-b10288-deberta-v1 && cmake -B build-fork && cmake --build build-fork -j"$(nproc)"
> ```
> For the jina and bge rerank fixes, the difference is numerical and needs a repaired GGUF, because
> the official conversions are missing tensors the fixes rely on:
> ```console
> python3 scripts/wikiq/fix-rerank-gguf.py <official-jina-gguf> <output-gguf>
> ./build-stock/bin/llama-server --rerank -m <output-gguf> --port 8080
> ./build-fork/bin/llama-server  --rerank -m <output-gguf> --port 8081
> ```
> Score the same query and documents against both and compare the raw scores and the resulting
> order against the model's own HF forward pass. Stock compresses the range and reorders; this
> fork does not. The four causes are itemised above, so a disagreement can be attributed rather
> than just observed.
>
> For DeBERTa there is no such pair to run, for the reason given above: stock produces no GGUF to
> serve. The eight `test-deberta-*` ctest targets are the check that runs here. Fidelity against
> the real HF forward pass is gated outside this repository, on a fixture of 8 premise-hypothesis
> pairs scored through `/v1/embeddings`: all three logits of every pair agree with the HF reference
> to within `5e-2`, with a largest observed gap of `~4.77e-5` across the 24 comparisons.
>
> **Pinning this fork.** Pin a **tag**. Tags here are immutable and namespaced `wikiq-<upstream
> base>-<what it adds>-v<n>`, because this fork inherited upstream's whole tag namespace and a bare
> `v1` would be indistinguishable from it. The current release is `wikiq-b10288-deberta-v1`, cut at
> a commit that passed the full conformance gate.
>
> **Branches move, and are not a supported pinning unit.** Work happens on one branch per upstream
> base, `wikiq-patches-b10288` today. Rebasing onto a newer upstream release creates a new branch
> beside it, `wikiq-patches-b1xxxx`, rather than force-pushing over this one, so an existing pin
> keeps resolving and the branch name tells you which upstream release you are on. The default
> branch follows the newest of these, so it changes too.
>
> See commit `d191d0f` for the full technical writeup of the jina-bert-v2 fixes, `f7127b2` for an
> unrelated interaction bug between PR #21729 and the DeepSeek-v4 code path, and `9c51923` for the
> bge-reranker-v2-m3 crash fix.
>
> This fork is publicly available so it can be pinned as a dependency. It is maintained for wikiq's
> use rather than as a source of upstream PRs: the work here is AI-assisted well beyond what
> [`AGENTS.md`](AGENTS.md) accepts in a contribution, and none of it is offered for upstream to
> maintain.

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev branches](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-features.md) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
