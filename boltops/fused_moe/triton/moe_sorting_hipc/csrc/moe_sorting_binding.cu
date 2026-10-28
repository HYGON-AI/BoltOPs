// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modified by Hygon Information Technology Co., Ltd., 2026.
// Upstream reference: https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/csrc/py_itfs_ck/moe_sorting_kernels.cu

#include <torch/all.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include "moe_sorting_types.h"

namespace {

void moe_sorting_fwd(torch::Tensor& topk_ids,
                     torch::Tensor& topk_weights,
                     torch::Tensor& sorted_token_ids,
                     torch::Tensor& sorted_weights,
                     torch::Tensor& sorted_expert_ids,
                     torch::Tensor& tokens_positions_per_expert,
                     torch::Tensor& num_valid_ids,
                     std::optional<torch::Tensor> moe_buf,
                     int num_experts,
                     int unit_size,
                     std::optional<torch::Tensor> local_expert_mask = std::nullopt)
{
    TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
                "topk_ids must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float,
                "topk_weights must be float32");

    const int num_tokens = topk_ids.size(0);
    const int topk       = topk_ids.size(1);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(topk_ids));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const int workspace_size =
        moe_sorting_get_workspace_size(num_tokens, num_experts, topk);
    void* workspace_ptr = nullptr;
    torch::Tensor workspace;
    if(workspace_size > 0)
    {
        // Keep the aiter allocation semantics exactly: workspace_size is a byte count,
        // while the tensor is int32. The over-allocation is intentional compatibility.
        workspace = torch::zeros(
            {workspace_size},
            torch::TensorOptions().dtype(topk_ids.dtype()).device(topk_ids.device()));
        workspace_ptr = workspace.data_ptr();
    }

    moe_sorting(
        {"int32", "fp32", local_expert_mask.has_value()},
        {topk_ids.data_ptr(),
         topk_weights.data_ptr(),
         local_expert_mask.has_value() ? local_expert_mask->data_ptr() : nullptr,
         sorted_token_ids.data_ptr(),
         sorted_weights.data_ptr(),
         sorted_expert_ids.data_ptr(),
         tokens_positions_per_expert.data_ptr(),
         num_valid_ids.data_ptr(),
         moe_buf.has_value() ? moe_buf->data_ptr() : nullptr,
         workspace_ptr,
         num_tokens,
         unit_size,
         num_experts,
         topk,
         moe_buf.has_value() ? static_cast<int64_t>(moe_buf->nbytes()) : 0},
        {stream});
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("moe_sorting_fwd",
               &moe_sorting_fwd,
               pybind11::arg("topk_ids"),
               pybind11::arg("topk_weights"),
               pybind11::arg("sorted_token_ids"),
               pybind11::arg("sorted_weights"),
               pybind11::arg("sorted_expert_ids"),
               pybind11::arg("tokens_positions_per_expert"),
               pybind11::arg("num_valid_ids"),
               pybind11::arg("moe_buf"),
               pybind11::arg("num_experts"),
               pybind11::arg("unit_size"),
               pybind11::arg("local_expert_mask") = std::nullopt);
}
