// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#define BOLT_OPS_HIPC_HOST inline __host__
#define BOLT_OPS_HIPC_DEVICE inline __device__
#define BOLT_OPS_HIPC_HOST_DEVICE inline __host__ __device__

#ifndef BOLT_OPS_HIPC_REFERENCE_MOE_SORTING_MOCK_ID
#define BOLT_OPS_HIPC_REFERENCE_MOE_SORTING_MOCK_ID 1
#endif
#ifndef BOLT_OPS_HIPC_REFERENCE_MOE_SORTING_MOCK_EXPERT_ID
#define BOLT_OPS_HIPC_REFERENCE_MOE_SORTING_MOCK_EXPERT_ID 0
#endif
#ifndef BOLT_OPS_HIPC_WA_ISSUE_2028
#define BOLT_OPS_HIPC_WA_ISSUE_2028 0
#endif

#define BOLT_OPS_HIP_CHECK(expr)                                                     \
    do {                                                                           \
        const hipError_t error = (expr);                                            \
        if(error != hipSuccess)                                                     \
            throw std::runtime_error(hipGetErrorString(error));                     \
    } while(false)

namespace boltops_hipc {

using index_t      = int32_t;
using long_index_t = int64_t;

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <index_t I>
using number = std::integral_constant<index_t, I>;

template <typename T, index_t N>
using ext_vector_t = T __attribute__((ext_vector_type(N)));

using uint8x16_t = uint8_t __attribute__((ext_vector_type(16)));

template <typename T, typename U>
BOLT_OPS_HIPC_HOST_DEVICE constexpr auto max(T x, U y)
{
    return x > y ? x : y;
}

template <typename T, typename U>
BOLT_OPS_HIPC_HOST_DEVICE constexpr auto min(T x, U y)
{
    return x < y ? x : y;
}

template <typename T, typename U>
BOLT_OPS_HIPC_HOST_DEVICE constexpr auto integer_divide_ceil(T x, U y)
{
    return (x + y - 1) / y;
}

template <typename... Ts>
BOLT_OPS_HIPC_HOST_DEVICE constexpr auto make_tuple(Ts&&... xs)
{
    return std::tuple<std::decay_t<Ts>...>{std::forward<Ts>(xs)...};
}

template <typename To, typename From>
BOLT_OPS_HIPC_HOST_DEVICE constexpr To bit_cast(const From& value)
{
    return __builtin_bit_cast(To, value);
}

template <index_t Begin, index_t End, index_t Step>
struct static_for
{
    template <typename F>
    BOLT_OPS_HIPC_HOST_DEVICE constexpr void operator()(F f) const
    {
        static_assert(Step > 0, "only positive static_for steps are used here");
        [&]<index_t... I>(std::integer_sequence<index_t, I...>) {
            (f(number<Begin + I * Step>{}), ...);
        }(std::make_integer_sequence<index_t, (End - Begin) / Step>{});
    }
};

BOLT_OPS_HIPC_HOST_DEVICE constexpr index_t get_smem_capacity()
{
#if defined(__gfx950__)
    return 163840;
#else
    return 65536;
#endif
}

struct mdiv
{
    uint32_t divisor;
    uint32_t multiplier;
    uint32_t shift;

    BOLT_OPS_HIPC_HOST_DEVICE mdiv() : divisor(0), multiplier(0), shift(0) {}

    BOLT_OPS_HIPC_HOST_DEVICE explicit mdiv(uint32_t divisor_) : divisor(divisor_)
    {
        shift = 0;
        while((1U << shift) < divisor)
            ++shift;
        const uint64_t tmp = (static_cast<uint64_t>((1UL << shift) - divisor)) << 32;
        multiplier        = static_cast<uint32_t>(tmp / divisor + 1);
    }

    BOLT_OPS_HIPC_HOST_DEVICE uint32_t div(uint32_t dividend) const
    {
#if defined(__HIP_DEVICE_COMPILE__)
        const uint32_t hi = __umulhi(dividend, multiplier);
#else
        const uint32_t hi = static_cast<uint32_t>(
            (static_cast<uint64_t>(dividend) * multiplier) >> 32);
#endif
        return (hi + dividend) >> shift;
    }

    BOLT_OPS_HIPC_HOST_DEVICE void
    divmod(uint32_t dividend, uint32_t& quotient, uint32_t& remainder) const
    {
        quotient  = div(dividend);
        remainder = dividend - quotient * divisor;
    }
};

struct stream_config
{
    hipStream_t stream_id_;
};

template <int MaxThreads, int MinBlocks, typename Kernel, typename... Args>
__launch_bounds__(MaxThreads, MinBlocks) __global__ void kentry(Args... args)
{
    Kernel{}(args...);
}

template <int MaxThreads = 768, int MinBlocks = 1, typename Kernel, typename... Args>
BOLT_OPS_HIPC_HOST auto
make_kernel(Kernel, dim3 grid, dim3 block, std::size_t lds_bytes, Args... args)
{
    const auto fn = kentry<MaxThreads, MinBlocks, Kernel, Args...>;
    return [=](const stream_config& stream) {
        hipLaunchKernelGGL(fn, grid, block, lds_bytes, stream.stream_id_, args...);
    };
}

template <typename... Callables>
BOLT_OPS_HIPC_HOST float launch_kernel(const stream_config& stream, Callables... calls)
{
    (calls(stream), ...);
    BOLT_OPS_HIP_CHECK(hipGetLastError());
    return 0.0f;
}

} // namespace boltops_hipc
