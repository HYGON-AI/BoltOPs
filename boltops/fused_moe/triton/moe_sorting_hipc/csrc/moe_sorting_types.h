// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// Modified by Hygon Information Technology Co., Ltd., 2026.
// Upstream reference: https://github.com/ROCm/composable_kernel/blob/baf508b98fce5a23ab6cee21ad3fc8822ecda125/example/ck_tile/13_moe_sorting/moe_sorting_api.hpp
#pragma once

#include "compat.h"

#include <string>

namespace boltops_hipc {

struct MoeSortingHostArgs
{
    const void* p_topk_ids;
    const void* p_weights;
    const void* p_local_expert_mask;
    void* p_sorted_token_ids;
    void* p_sorted_weights;
    void* p_sorted_expert_ids;
    void* p_tokens_positions_per_expert;
    void* p_total_tokens_post_pad;
    void* p_moe_buf;
    void* p_ws;
    index_t tokens;
    index_t unit_size;
    index_t num_experts;
    index_t topk;
    long_index_t moe_buf_bytes;
};

} // namespace boltops_hipc

struct moe_sorting_trait
{
    std::string index_type;
    std::string weight_type;
    bool local_expert_masking;
};

struct moe_sorting_args : public boltops_hipc::MoeSortingHostArgs
{
};

int moe_sorting_get_workspace_size(int tokens, int num_experts, int topk);
float moe_sorting(moe_sorting_trait t,
                  moe_sorting_args a,
                  boltops_hipc::stream_config stream);
float moe_sorting_mp(moe_sorting_trait t,
                     moe_sorting_args a,
                     boltops_hipc::stream_config stream);

