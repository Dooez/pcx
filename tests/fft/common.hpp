#pragma once
#include "pcx/include/fft_impl.hpp"

#include <cmath>
#include <print>
#include <vector>

namespace pcx::testing {
#ifdef FULL_FFT_TEST
inline constexpr auto f32_widths = uZ_seq<4, 8, 16>{};
inline constexpr auto f64_widths = uZ_seq<2, 4, 8>{};
inline constexpr auto half_tw    = meta::val_seq<false, true>{};
inline constexpr auto low_k      = meta::val_seq<false, true>{};
inline constexpr auto node_sizes = uZ_seq<2, 4, 8, 16>{};

#else
inline constexpr auto f32_widths = uZ_seq<16>{};
inline constexpr auto f64_widths = uZ_seq<8>{};
inline constexpr auto half_tw    = meta::val_seq<true>{};
inline constexpr auto low_k      = meta::val_seq<true>{};
inline constexpr auto node_sizes = uZ_seq<8>{};
#endif
inline constexpr auto local_tw = meta::val_seq<false>{};

template<typename T, uZ Width>
bool test_fft(const std::vector<std::complex<T>>& signal,
              const std::vector<std::complex<T>>& chk_fwd,
              const std::vector<std::complex<T>>& chk_rev,
              std::vector<std::complex<T>>&       s1,
              std::vector<T>&                     twvec);
template<typename fX>
using std_vec2d = std::vector<std::vector<std::complex<fX>>>;
template<typename fX, uZ Width>
bool test_par(const std_vec2d<fX>&                 signal,
              std_vec2d<fX>&                       s1,
              const std::vector<std::complex<fX>>& chk_fwd,
              const std::vector<std::complex<fX>>& chk_rev,
              std::vector<fX>&                     twvec,
              bool                                 local_check,
              bool                                 fwd,
              bool                                 rev,
              bool                                 inplace,
              bool                                 external);

}    // namespace pcx::testing

template<typename T>
struct std::formatter<std::complex<T>, char> {
    std::formatter<T, char> m_fmt;
    template<class ParseContext>
    constexpr auto parse(ParseContext& ctx) -> ParseContext::iterator {
        return m_fmt.parse(ctx);
    }

    template<class FmtContext>
    FmtContext::iterator format(std::complex<T> cxv, FmtContext& ctx) const {
        *ctx.out() = "(";
        auto out   = m_fmt.format(cxv.real(), ctx);
        *(out++)   = ",";
        *out       = " ";
        out        = m_fmt.format(cxv.imag(), ctx);
        *(out++)   = ")";
        return out;
    }
};
static auto get_type_name(pcx::meta::types<pcx::f32>) {
    return std::string_view("f32");
}
static auto get_type_name(pcx::meta::types<pcx::f64>) {
    return std::string_view("f64");
}
template<typename T>
struct std::formatter<pcx::meta::types<T>> {
    template<class ParseContext>
    constexpr auto parse(ParseContext& ctx) -> ParseContext::iterator {
        auto it = ctx.begin();
        if (it == ctx.end())
            return it;
        if (it != ctx.end() && *it != '}')
            throw std::format_error("Invalid format args for pcx::meta::types<T>.");

        return it;
    }

    template<class FmtContext>
    FmtContext::iterator format(pcx::meta::types<T> mt, FmtContext& ctx) const {
        return std::ranges::copy(get_type_name(mt), ctx.out()).out;
    }
};

namespace pcx::testing {
template<typename fX>
void naive_fft(std::vector<std::complex<fX>>& data, uZ node_size, uZ vec_width);
template<typename fX>
void naive_reverse(std::vector<std::complex<fX>>& data, uZ node_size, uZ vec_width);

inline auto check_nan(const std::vector<std::complex<f32>>& vec) {
    return std::ranges::any_of(vec, [](auto v) { return std::isnan(v.real()) || std::isnan(v.imag()); });
};
inline auto check_nan(const std::vector<std::complex<f64>>& vec) {
    return std::ranges::any_of(vec, [](auto v) { return std::isnan(v.real()) || std::isnan(v.imag()); });
};
template<typename fX>
bool check_correctness(const std::vector<std::complex<fX>>& naive,
                       const std::vector<std::complex<fX>>& pcx,
                       uZ                                   width,
                       uZ                                   node_size,
                       bool                                 lowk,
                       bool                                 local_tw,
                       bool                                 half_tw);

template<typename fX>
bool par_check_correctness(std::complex<fX>                     val,
                           const std::vector<std::complex<fX>>& pcx,
                           uZ                                   fft_size,
                           uZ                                   fft_id,
                           uZ                                   width,
                           uZ                                   node_size,
                           bool                                 local_tw);
template<typename fX>
bool par_test_proto(auto                                 node_size,
                    auto                                 width,
                    auto                                 lowk,
                    auto                                 local_tw,
                    const std_vec2d<fX>&                 signal,
                    std_vec2d<fX>&                       s1,
                    const std::vector<std::complex<fX>>& chk_fwd,
                    const std::vector<std::complex<fX>>& chk_rev,
                    std::vector<fX>&                     twvec,
                    bool                                 local_check,
                    bool                                 fwd,
                    bool                                 rev,
                    bool                                 inplace,
                    bool                                 external) {
    constexpr auto half_tw  = std::true_type{};
    constexpr auto coherent = std::false_type{};

    auto fft_size  = signal.size();
    auto data_size = signal[0].size();

    using fimpl      = pcx::detail_::transform<node_size, fX, width>;
    using data_t     = pcx::detail_::data_info<fX, false, std_vec2d<fX>>;
    using src_data_t = pcx::detail_::data_info<const fX, false, const std_vec2d<fX>>;

    s1            = signal;
    auto s1_info  = data_t{.data_ptr = &s1};
    auto src_info = src_data_t{.data_ptr = &signal};

    auto tw = [&] {
        using tw_t = detail_::tw_data_t<fX, local_tw>;
        if constexpr (local_tw) {
            return tw_t{1, 0};
        } else {
            twvec.resize(0);
            fimpl::insert_tw(twvec, fft_size, lowk, half_tw, coherent);
            return tw_t{twvec.data()};
        }
    }();
    auto tw_rev = [&] {
        using tw_t = detail_::tw_data_t<fX, local_tw>;
        if constexpr (local_tw) {
            return tw_t{fft_size, 0};
        } else {
            return tw_t{&(*twvec.end())};
        }
    }();

    constexpr auto pck_dst = cxpack<1, fX>{};
    constexpr auto pck_src = cxpack<1, fX>{};

    auto l_chk_fwd = std::vector<std::complex<fX>>{};
    auto l_chk_rev = std::vector<std::complex<fX>>{};

    if (local_check) {
        l_chk_fwd = chk_fwd;
        l_chk_rev = chk_rev;
        naive_fft(l_chk_fwd, node_size, width);
        naive_reverse(l_chk_rev, node_size, width);
    }
    auto run_check = [&](bool fwd) {
        if (local_check) {
            if (fwd) {
                for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1, l_chk_fwd)) {
                    if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                        return false;
                }
                return true;
            }
            for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1, l_chk_rev)) {
                if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                    return false;
            }
            return true;
        }
        if (fwd) {
            for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1, chk_fwd)) {
                if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                    return false;
            }
            return true;
        }
        for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1, chk_rev)) {
            if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                return false;
        }
        return true;
    };

    if (inplace && fwd) {
        std::print("[Inplace fwd    ]");
        fimpl::perform(pck_dst,
                       pck_src,
                       half_tw,
                       lowk,
                       s1_info,
                       detail_::inplace_src,
                       fft_size,
                       tw,
                       detail_::blank_sorter,
                       data_size);
        if (!run_check(true))
            return false;
        std::println("[Success] {}×{}, width {}, node size {}{}.",
                     pcx::meta::types<fX>{},
                     fft_size,
                     width.value,
                     node_size.value,
                     local_tw ? ", local tw" : "");
    }
    if (inplace && rev) {
        s1 = signal;
        std::print("[Inplace rev    ]");
        fimpl::perform_rev(pck_dst,
                           pck_src,
                           half_tw,
                           lowk,
                           s1_info,
                           detail_::inplace_src,
                           fft_size,
                           tw_rev,
                           data_size);
        if (!run_check(false))
            return false;
        std::println("[Success] {}×{}, width {}, node size {}{}.",
                     pcx::meta::types<fX>{},
                     fft_size,
                     width.value,
                     node_size.value,
                     local_tw ? ", local tw" : "");
    }
    if (external && fwd) {
        std::print("[Externl fwd    ]");
        fimpl::perform(pck_dst,
                       pck_src,
                       half_tw,
                       lowk,
                       s1_info,
                       src_info,
                       fft_size,
                       tw,
                       detail_::blank_sorter,
                       data_size);
        if (!run_check(true))
            return false;
        std::println("[Success] {}×{}, width {}, node size {}{}.",
                     pcx::meta::types<fX>{},
                     fft_size,
                     width.value,
                     node_size.value,
                     local_tw ? ", local tw" : "");
    }
    if (external && rev) {
        std::print("[Externl rev    ]");
        fimpl::perform_rev(pck_dst, pck_src, half_tw, lowk, s1_info, src_info, fft_size, tw_rev, data_size);
        if (!run_check(false))
            return false;
        std::println("[Success] {}×{}, width {}, node size {}{}.",
                     pcx::meta::types<fX>{},
                     fft_size,
                     width.value,
                     node_size.value,
                     local_tw ? ", local tw" : "");
    }

    return true;
}

template<typename fX, uZ Width, uZ NodeSize, bool LowK, bool LocalTw, bool HalfTw>
bool test_prototype(const std::vector<std::complex<fX>>& signal,
                    const std::vector<std::complex<fX>>& chk_fwd,
                    const std::vector<std::complex<fX>>& chk_rev,
                    std::vector<std::complex<fX>>&       s1,
                    std::vector<fX>&                     twvec,
                    bool                                 local_check = false,
                    bool                                 fwd         = true,
                    bool                                 rev         = true,
                    bool                                 inplace     = true,
                    bool                                 ext         = true) {
    constexpr auto half_tw = std::bool_constant<HalfTw>{};
    constexpr auto lowk    = std::bool_constant<LowK>{};

    auto fft_size = signal.size();
    s1            = signal;

    using fimpl = pcx::detail_::transform<NodeSize, fX, Width>;

    auto* data_ptr = reinterpret_cast<fX*>(s1.data());

    auto tw = [&] {
        using tw_t = detail_::tw_data_t<fX, LocalTw>;
        if constexpr (LocalTw) {
            return tw_t{1, 0};
        } else {
            twvec.resize(0);
            fimpl::insert_tw(twvec, fft_size, lowk, half_tw, std::true_type{});
            return tw_t{twvec.data()};
        }
    }();
    auto tw_rev = [&] {
        using tw_t = detail_::tw_data_t<fX, LocalTw>;
        if constexpr (LocalTw) {
            return tw_t{fft_size, 0};
        } else {
            return tw_t{&(*twvec.end())};
        }
    }();

    constexpr auto pck_dst   = cxpack<1, fX>{};
    constexpr auto pck_src   = cxpack<1, fX>{};
    auto           l_chk_fwd = std::vector<std::complex<fX>>{};
    auto           l_chk_rev = std::vector<std::complex<fX>>{};

    if (local_check) {
        l_chk_fwd = signal;
        l_chk_rev = signal;
        naive_fft(l_chk_fwd, NodeSize, Width);
        naive_reverse(l_chk_rev, NodeSize, Width);
    }
    auto run_check = [&](bool fwd) {
        if (local_check) {
            if (fwd)
                return check_correctness(l_chk_fwd, s1, Width, NodeSize, LowK, LocalTw, half_tw);
            else
                return check_correctness(l_chk_rev, s1, Width, NodeSize, LowK, LocalTw, half_tw);
        }
        if (fwd)
            return check_correctness(chk_fwd, s1, Width, NodeSize, LowK, LocalTw, half_tw);
        else
            return check_correctness(chk_rev, s1, Width, NodeSize, LowK, LocalTw, half_tw);
    };

    using dst_info_t = detail_::sequential_data_info<fX>;
    auto s1_info     = dst_info_t{{}, data_ptr};
    // auto src_info    = detail_::data_info<const fX, true>{reinterpret_cast<const fX*>(data_ptr)};
    auto src_info = detail_::sequential_data_info<const fX>{{}, reinterpret_cast<const fX*>(signal.data())};


    if (inplace && fwd) {
        std::print("[inplace fwd    ]");
        s1 = signal;
        fimpl::perform(pck_dst,
                       pck_src,
                       half_tw,
                       lowk,
                       s1_info,
                       detail_::inplace_src,
                       fft_size,
                       tw,
                       detail_::blank_sorter);
        if (!run_check(true))
            return false;
    }
    if (inplace && rev) {
        std::print("[inplace rev    ]");
        s1 = signal;
        fimpl::perform_rev(pck_dst, pck_src, half_tw, lowk, s1_info, detail_::inplace_src, fft_size, tw_rev);
        if (!run_check(false))
            return false;
    }

    if (ext && fwd) {
        std::print("[externl fwd    ]");
        // s1 = signal;
        stdr::fill(s1, -69.);
        fimpl::perform(pck_dst,
                       pck_src,
                       half_tw,
                       lowk,
                       s1_info,
                       src_info,
                       fft_size,
                       tw,
                       detail_::blank_sorter);
        if (!run_check(true))
            return false;
    }
    if (ext && rev) {
        std::print("[externl rev    ]");
        // s1 = signal;
        stdr::fill(s1, -69.);
        fimpl::perform_rev(pck_dst, pck_src, half_tw, lowk, s1_info, src_info, fft_size, tw_rev);
        if (!run_check(false))
            return false;
    }

    s1              = signal;
    using fimpl_coh = pcx::detail_::sequential_subtransform<NodeSize, fX, Width>;
    auto coh_align  = fimpl_coh::get_align_node(fft_size);
    auto tw_coh_fwd = [&] {
        using tw_t = detail_::tw_data_t<fX, LocalTw>;
        if constexpr (LocalTw) {
            return tw_t{1, 0};
        } else {
            twvec.resize(0);
            [&]<uZ... Is>(uZ_seq<Is...>) {
                auto check_align = [&]<uZ I>(uZ_ce<I>) {
                    constexpr auto l_node_size = detail_::powi(2, I);
                    if (l_node_size != coh_align)
                        return false;
                    auto tw_data = detail_::tw_data_t<fX, true>{1, 0};
                    fimpl_coh::insert_tw(twvec,
                                         detail_::align_param<l_node_size, true>{},
                                         lowk,
                                         fft_size,
                                         tw_data,
                                         half_tw);
                    return true;
                };
                (void)(check_align(uZ_ce<Is>{}) || ...);
            }(make_uZ_seq<detail_::log2i(NodeSize)>{});
            return tw_t{twvec.data()};
        }
    }();
    auto tw_coh_rev = [&] {
        using tw_t = detail_::tw_data_t<fX, LocalTw>;
        if constexpr (LocalTw) {
            return tw_t{fft_size, 0};
        } else {
            return tw_t{&(*twvec.end())};
        }
    }();
    if (fwd) {
        std::print("[inplace fwd coh]");
        s1            = signal;
        auto tw_coh_c = tw_coh_fwd;
        fimpl_coh::perform(pck_dst,
                           pck_src,
                           lowk,
                           half_tw,
                           std::false_type{},    //reverse
                           std::false_type{},    //conj_tw
                           fft_size,
                           s1_info.mul_stride(Width),
                           detail_::inplace_src,
                           coh_align,
                           tw_coh_c);
        if (!run_check(true))
            return false;
    }
    if (rev) {
        std::print("[inplace rev coh]");
        s1            = signal;
        auto tw_coh_c = tw_coh_rev;
        fimpl_coh::perform(pck_dst,
                           pck_src,
                           lowk,
                           half_tw,
                           std::true_type{},    // reverse
                           std::true_type{},    // conj_tw
                           fft_size,
                           s1_info.mul_stride(Width),
                           detail_::inplace_src,
                           coh_align,
                           tw_coh_c);
        if (!run_check(false))
            return false;
    }

    // std::print("[eCoherent]");
    // tw_coh_c = tw_coh;
    // fimpl_coh::perform(pck_dst, pck_src, lowk, half_tw, fft_size, s1_info, src_info, coh_align, tw_coh_c);
    // if (!run_check())
    //     return false;

    return true;
}

template<typename fX, uZ VecWidth, uZ... NodeSize, bool... LowK, bool... LocalTw>
bool par_run_tests(uZ_seq<NodeSize...>,
                   meta::val_seq<LowK...>,
                   meta::val_seq<LocalTw...>,
                   const std_vec2d<fX>&                 signal,
                   std_vec2d<fX>&                       s1,
                   const std::vector<std::complex<fX>>& chk_fwd,
                   const std::vector<std::complex<fX>>& chk_rev,
                   std::vector<fX>&                     twvec,
                   bool                                 local_check,
                   bool                                 fwd,
                   bool                                 rev,
                   bool                                 inplace,
                   bool                                 external) {
    auto lk_passed = [&](auto lowk) {
        auto ltw_passed = [&](auto local_tw) {
            return ((signal.size() <= NodeSize
                     || par_test_proto(uZ_ce<NodeSize>{},
                                       uZ_ce<VecWidth>{},
                                       lowk,
                                       local_tw,
                                       signal,
                                       s1,
                                       chk_fwd,
                                       chk_rev,
                                       twvec,
                                       local_check,
                                       fwd,
                                       rev,
                                       inplace,
                                       external))
                    && ...);
        };
        return (ltw_passed(val_ce<LocalTw>{}) && ...);
    };
    return (lk_passed(val_ce<LowK>{}) && ...);
}

template<typename fX, uZ VecWidth, uZ... NodeSize, bool... low_k, bool... local_tw, bool... half_tw>
bool run_tests(uZ_seq<NodeSize...>,
               meta::val_seq<low_k...>,
               meta::val_seq<local_tw...>,
               meta::val_seq<half_tw...>,
               const std::vector<std::complex<fX>>& signal,
               const std::vector<std::complex<fX>>& chk_fwd,
               const std::vector<std::complex<fX>>& chk_rev,
               std::vector<std::complex<fX>>&       s1,
               std::vector<fX>&                     twvec) {
    auto lk_passed = [&]<bool LowK>(val_ce<LowK>) {
        auto ltw_passed = [&]<bool LocalTw>(val_ce<LocalTw>) {
            auto htw_passed = [&]<bool HalfTw>(val_ce<HalfTw>) {
                return ((signal.size() <= NodeSize * VecWidth
                         || test_prototype<fX, VecWidth, NodeSize, LowK, LocalTw, HalfTw>(signal,
                                                                                          chk_fwd,
                                                                                          chk_rev,
                                                                                          s1,
                                                                                          twvec))
                        && ...);
            };
            return (htw_passed(val_ce<half_tw>{}) && ...);
        };
        return (ltw_passed(val_ce<local_tw>{}) && ...);
    };
    return (lk_passed(val_ce<low_k>{}) && ...);
};

}    // namespace pcx::testing
