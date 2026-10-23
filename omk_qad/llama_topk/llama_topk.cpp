// SPDX-License-Identifier: Apache-2.0
// Teacher top-K over given token sequences with libllama (a GGUF teacher for omk_qad --phase cache).
//
// llama-server returns log-probs of generated tokens only; the cache needs the teacher's distribution at every
// position of a sequence it is GIVEN. This process loads the model once and serves sequences over stdin/stdout:
//
//   in : int32 n, int32 ids[n], uint8 want[n - 1]            (n == 0 ends the process)
//   out: for each position i < n - 1: int32 idx[K], float32 logprob[K], float32 logprob of ids[i + 1]
//        (positions with want[i] == 0 are not computed: idx = 0..K-1, logprob = -log(n_vocab), target = 0)
//
// The log-probs are those of the model's own logits (llama.cpp applies Gemma-4's final-logit soft cap in the
// graph). Build: see omk_qad/gguf_teacher.py.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static bool read_all(void * p, size_t n) { return fread(p, 1, n, stdin) == n; }

static void write_all(const void * p, size_t n) {
    if (fwrite(p, 1, n, stdout) != n) {
        fprintf(stderr, "llama_topk: write failed\n");
        exit(1);
    }
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_ctx = 16384, topk = 64, ngl = 999, n_ubatch = 512, threads = 16;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--model") model_path = v;
        else if (k == "--ctx") n_ctx = std::stoi(v);
        else if (k == "--topk") topk = std::stoi(v);
        else if (k == "--ngl") ngl = std::stoi(v);
        else if (k == "--ubatch") n_ubatch = std::stoi(v);
        else if (k == "--threads") threads = std::stoi(v);
        else { fprintf(stderr, "llama_topk: unknown option %s\n", k.c_str()); return 2; }
    }
    if (model_path.empty()) { fprintf(stderr, "llama_topk: --model is required\n"); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "llama_topk: cannot load %s\n", model_path.c_str()); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ubatch;
    cp.n_ubatch = n_ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "llama_topk: cannot create a context of %d\n", n_ctx); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(stderr, "LLAMA_TOPK_READY n_vocab=%d n_ctx=%d topk=%d\n", n_vocab, n_ctx, topk);
    fflush(stderr);

    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);
    std::vector<int32_t> ids;
    std::vector<uint8_t> want;
    std::vector<int32_t> oidx;
    std::vector<float> oval, otgt;
    const float flat = -logf((float) n_vocab);
    for (;;) {
        int32_t n = 0;
        if (!read_all(&n, 4) || n == 0) break;
        if (n < 2 || n > n_ctx) { fprintf(stderr, "llama_topk: sequence of %d tokens (ctx %d)\n", n, n_ctx); return 1; }
        ids.resize(n);
        want.resize(n - 1);
        if (!read_all(ids.data(), 4 * (size_t) n) || !read_all(want.data(), n - 1)) return 1;
        oidx.assign((size_t) (n - 1) * topk, 0);
        oval.assign((size_t) (n - 1) * topk, flat);
        otgt.assign(n - 1, 0.0f);
        for (int i = 0; i < n - 1; i++)
            for (int k = 0; k < topk; k++) oidx[(size_t) i * topk + k] = k;
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int c0 = 0; c0 < n; c0 += n_ubatch) {
            const int c1 = std::min(n, c0 + n_ubatch);
            batch.n_tokens = 0;
            for (int i = c0; i < c1; i++) {
                const int j = batch.n_tokens++;
                batch.token[j] = ids[i];
                batch.pos[j] = i;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = i < n - 1 && want[i];
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_topk: decode failed at %d\n", c0); return 1; }
#pragma omp parallel num_threads(threads)
            {
                std::vector<int32_t> order(n_vocab);  // per thread
#pragma omp for schedule(dynamic, 8)
                for (int i = c0; i < c1; i++) {
                    if (!(i < n - 1 && want[i])) continue;
                    const float * lg = llama_get_logits_ith(ctx, i - c0);
                    float mx = -INFINITY;
                    for (int v = 0; v < n_vocab; v++) mx = std::max(mx, lg[v]);
                    double s = 0;
                    for (int v = 0; v < n_vocab; v++) s += exp((double) lg[v] - mx);
                    const float lse = mx + (float) log(s);
                    std::iota(order.begin(), order.end(), 0);
                    auto gt = [lg](int a, int b) { return lg[a] > lg[b]; };
                    std::nth_element(order.begin(), order.begin() + topk, order.end(), gt);
                    std::sort(order.begin(), order.begin() + topk, gt);
                    for (int k = 0; k < topk; k++) {
                        oidx[(size_t) i * topk + k] = order[k];
                        oval[(size_t) i * topk + k] = lg[order[k]] - lse;
                    }
                    otgt[i] = lg[ids[i + 1]] - lse;
                }
            }
        }
        for (int i = 0; i < n - 1; i++) {
            write_all(&oidx[(size_t) i * topk], 4 * (size_t) topk);
            write_all(&oval[(size_t) i * topk], 4 * (size_t) topk);
            write_all(&otgt[i], 4);
        }
        fflush(stdout);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
