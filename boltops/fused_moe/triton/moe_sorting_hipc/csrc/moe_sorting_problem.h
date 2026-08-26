// SPDX-License-Identifier: MIT
#pragma once

#include "compat.h"

namespace boltops_hipc {

template <typename IndexType_,
          typename WeightType_,
          index_t SubTokenTile_,
          bool SubTokenOneShot_,
          bool LocalExpertMasking_,
          bool SkipExpertsWithZeroTokens_ = true,
          index_t ExpertTile_             = 0>
struct MoeSortingProblemEx
{
    using WeightType = remove_cvref_t<WeightType_>;
    using IndexType  = remove_cvref_t<IndexType_>;

    static constexpr index_t WarpSize               = get_warp_size();
    static constexpr index_t WarpsPerBlock          = 1;
    static constexpr index_t SubTokenTile           = SubTokenTile_;
    static constexpr bool SubTokenOneShot           = SubTokenOneShot_;
    static constexpr bool LocalExpertMasking        = LocalExpertMasking_;
    static constexpr bool SkipExpertsWithZeroTokens = SkipExpertsWithZeroTokens_;
    static constexpr index_t ExpertTile             = ExpertTile_;
    static_assert(SubTokenTile == 1 || SubTokenTile == 2 || SubTokenTile == 4 ||
                  SubTokenTile == 8);
};

template <typename IndexType_,
          typename WeightType_,
          typename MeshType_,
          index_t SubTokenTile_,
          bool LocalExpertMasking_,
          bool SkipExpertsWithZeroTokens_ = true>
struct MoeSortingProblemMp
{
    using WeightType = remove_cvref_t<WeightType_>;
    using MeshType   = remove_cvref_t<MeshType_>;
    using IndexType  = remove_cvref_t<IndexType_>;

    static constexpr index_t SubTokenTile           = SubTokenTile_;
    static constexpr bool LocalExpertMasking        = LocalExpertMasking_;
    static constexpr bool SkipExpertsWithZeroTokens = SkipExpertsWithZeroTokens_;
    static_assert(SubTokenTile == 1 || SubTokenTile == 2 || SubTokenTile == 4 ||
                  SubTokenTile == 8 || SubTokenTile == 16);
};

} // namespace boltops_hipc

