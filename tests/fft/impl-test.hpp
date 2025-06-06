#include "common.hpp"

#include <cstring>
#include <generator>

namespace pcx::testing {
template<typename T, uZ Width, uZ NodeSize>
bool test_seq(const std::vector<std::complex<T>>& signal,
              const chk_t<T>&                     chk_fwd,
              const chk_t<T>&                     chk_rev,
              std::vector<std::complex<T>>&       s1,
              std::vector<T>&                     twvec,
              bool                                local_check,
              bool                                fwd,
              bool                                rev,
              bool                                inplace,
              bool                                external);
template<typename fX>
using std_vec2d = std::vector<std::vector<std::complex<fX>>>;
template<typename fX, uZ Width, uZ NodeSize>
bool test_par(const std_vec2d<fX>& signal,
              std_vec2d<fX>&       s1,
              const chk_t<fX>&     chk_fwd,
              const chk_t<fX>&     chk_rev,
              std::vector<fX>&     twvec,
              bool                 local_check,
              bool                 fwd,
              bool                 rev,
              bool                 inplace,
              bool                 external);
template<typename fX>
using parc_data = detail_::data_info<fX, true>;
template<typename fX, uZ Width, uZ NodeSize>
bool test_parc(parc_data<const fX> signal,
               parc_data<fX>       s1,
               uZ                  data_size,
               const chk_t<fX>&    chk_fwd,
               const chk_t<fX>&    chk_rev,
               std::vector<fX>&    twvec,
               bool                local_check,
               bool                fwd,
               bool                rev,
               bool                inplace,
               bool                external);

template<typename fX>
bool parc_test_proto(auto                node_size,
                     auto                width,
                     auto                lowk,
                     auto                local_tw,
                     auto                perm_type,
                     parc_data<const fX> signal_data,
                     parc_data<fX>       s1_data,
                     uZ                  data_size,
                     const chk_t<fX>&    chk_fwd,
                     const chk_t<fX>&    chk_rev,
                     std::vector<fX>&    twvec,
                     bool                local_check,
                     bool                fwd,
                     bool                rev,
                     bool                inplace,
                     bool                external) {
    constexpr auto half_tw    = std::true_type{};
    constexpr auto sequential = std::false_type{};
    constexpr auto pck_dst    = cxpack<1, fX>{};
    constexpr auto pck_src    = cxpack<1, fX>{};

    auto fft_size = chk_fwd(permute_t::bit_reversed).size();

    using fimpl = pcx::detail_::transform<node_size, fX, width>;

    auto s1 = [&](uZ i = 0) -> std::generator<std::span<std::complex<fX>>> {
        while (true) {
            auto ptr = reinterpret_cast<std::complex<fX>*>(s1_data.get_batch_base(i++));
            co_yield {ptr, data_size};
        }
    };
    auto reset_s1 = [&] {
        for (auto i: stdv::iota(0U, fft_size)) {
            auto* src = signal_data.get_batch_base(i);
            auto* dst = s1_data.get_batch_base(i);
            std::memcpy(dst, src, data_size * 2 * sizeof(fX));
        }
    };
    auto tw = [&] {
        using tw_t = detail_::tw_data_t<fX, local_tw>;
        if constexpr (local_tw) {
            return tw_t{1, 0};
        } else {
            twvec.resize(0);
            fimpl::insert_tw_tf(twvec, fft_size, lowk, half_tw, sequential);
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

    auto l_chk_fwd = std::vector<std::complex<fX>>{};
    auto l_chk_rev = std::vector<std::complex<fX>>{};
    auto chk_fwd_  = chk_fwd(perm_type);
    auto chk_rev_  = chk_rev(perm_type);

    auto permute_idxs = std::vector<u32>{};

    auto permute = [=] {
        if constexpr (perm_type == permute_t::bit_reversed) {
            return detail_::identity_permuter_t{};
        } else if constexpr (perm_type == permute_t::normal) {
            return detail_::br_permuter<node_size>{};
        } else if constexpr (perm_type == permute_t::shifted) {
            constexpr auto shifted_node = std::max({uZ(node_size), 4UZ});
            return detail_::br_permuter_shifted<shifted_node>{};
        }
    }();
    auto rev_permute = permute;
    if constexpr (perm_type != permute_t::bit_reversed) {
        using permuter_t = decltype(permute);
        if constexpr (perm_type == permute_t::normal) {
            constexpr auto coherent_size = 8194 / sizeof(fX);
            constexpr auto lane_size     = uZ_ce<std::max(64 / sizeof(fX) / 2, uZ(width))>{};

            auto coh_size = coherent_size / lane_size;
            permute       = permuter_t::insert_indexes(permute_idxs, fft_size, coh_size);
        } else {
            permute = permuter_t::insert_indexes(permute_idxs, fft_size);
        }
        rev_permute         = permute;
        permute.idx_ptr     = permute_idxs.data();
        rev_permute.idx_ptr = &(*permute_idxs.end());
    }
    constexpr auto perm_fmt = [=] {
        if constexpr (perm_type == permute_t::bit_reversed) {
            return "[BitRev]";
        } else if constexpr (perm_type == permute_t::normal) {
            return "[Normal]";
        } else if constexpr (perm_type == permute_t::shifted) {
            return "[Shiftd]";
        }
    }();
    if (local_check) {
        l_chk_fwd = chk_fwd_;
        l_chk_rev = chk_rev_;
        naive_fft(l_chk_fwd, node_size, width);
        naive_reverse(l_chk_rev, node_size, width);
    }

    auto run_check = [&](bool fwd) {
        if (local_check) {
            if (fwd) {
                for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1(), l_chk_fwd)) {
                    if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                        return false;
                }
                return true;
            }
            for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1(), l_chk_rev)) {
                if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                    return false;
            }
            return true;
        }
        if (fwd) {
            for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1(), chk_fwd_)) {
                if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                    return false;
            }
            return true;
        }
        for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1(), chk_rev_)) {
            if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                return false;
        }
        return true;
    };

    if (inplace && fwd) {
        reset_s1();
        std::print("[Inp][Fwd][Parc]");
        std::print(perm_fmt);
        fimpl::perform_auto_size(pck_dst,
                                 pck_src,
                                 half_tw,
                                 lowk,
                                 s1_data,
                                 detail_::inplace_src,
                                 fft_size,
                                 tw,
                                 permute,
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
        reset_s1();
        std::print("[Inp][Rev][Parc]");
        std::print(perm_fmt);
        fimpl::perform_rev_auto_size(pck_dst,
                                     pck_src,
                                     half_tw,
                                     lowk,
                                     s1_data,
                                     detail_::inplace_src,
                                     fft_size,
                                     tw_rev,
                                     rev_permute,
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
        std::print("[Ext][Fwd][Parc]");
        std::print(perm_fmt);
        fimpl::perform_auto_size(pck_dst,
                                 pck_src,
                                 half_tw,
                                 lowk,
                                 s1_data,
                                 signal_data,
                                 fft_size,
                                 tw,
                                 permute,
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
        std::print("[Ext][Rev][Parc]");
        std::print(perm_fmt);
        fimpl::perform_rev_auto_size(pck_dst,
                                     pck_src,
                                     half_tw,
                                     lowk,
                                     s1_data,
                                     signal_data,
                                     fft_size,
                                     tw_rev,
                                     rev_permute,
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
    return true;
};
template<typename fX>
bool par_test_proto(auto                 node_size,
                    auto                 width,
                    auto                 lowk,
                    auto                 local_tw,
                    auto                 perm_type,
                    const std_vec2d<fX>& signal,
                    std_vec2d<fX>&       s1,
                    const chk_t<fX>&     chk_fwd,
                    const chk_t<fX>&     chk_rev,
                    std::vector<fX>&     twvec,
                    bool                 local_check,
                    bool                 fwd,
                    bool                 rev,
                    bool                 inplace,
                    bool                 external) {
    constexpr auto half_tw    = std::true_type{};
    constexpr auto sequential = std::false_type{};

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
            fimpl::insert_tw_tf(twvec, fft_size, lowk, half_tw, sequential);
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
    auto chk_fwd_  = chk_fwd(perm_type);
    auto chk_rev_  = chk_rev(perm_type);

    auto permute_idxs = std::vector<u32>{};

    auto permute = [=] {
        if constexpr (perm_type == permute_t::bit_reversed) {
            return detail_::identity_permuter_t{};
        } else if constexpr (perm_type == permute_t::normal) {
            return detail_::br_permuter<node_size>{};
        } else if constexpr (perm_type == permute_t::shifted) {
            constexpr auto shifted_node = std::max({uZ(node_size), 4UZ});
            return detail_::br_permuter_shifted<shifted_node>{};
        }
    }();
    auto rev_permute = permute;
    if constexpr (perm_type != permute_t::bit_reversed) {
        using permuter_t = decltype(permute);
        if constexpr (perm_type == permute_t::normal) {
            constexpr auto coherent_size = 8194 / sizeof(fX);
            constexpr auto lane_size     = uZ_ce<std::max(64 / sizeof(fX) / 2, uZ(width))>{};

            auto coh_size = coherent_size / lane_size;
            permute       = permuter_t::insert_indexes(permute_idxs, fft_size, coh_size);
        } else {
            permute = permuter_t::insert_indexes(permute_idxs, fft_size);
        }
        rev_permute         = permute;
        permute.idx_ptr     = permute_idxs.data();
        rev_permute.idx_ptr = &(*permute_idxs.end());
    }
    constexpr auto perm_fmt = [=] {
        if constexpr (perm_type == permute_t::bit_reversed) {
            return "[BitRev]";
        } else if constexpr (perm_type == permute_t::normal) {
            return "[Normal]";
        } else if constexpr (perm_type == permute_t::shifted) {
            return "[Shiftd]";
        }
    }();
    if (local_check) {
        l_chk_fwd = chk_fwd_;
        l_chk_rev = chk_rev_;
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
            for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1, chk_fwd_)) {
                if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                    return false;
            }
            return true;
        }
        for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), s1, chk_rev_)) {
            if (!par_check_correctness(check_v, sv, fft_size, i, width, node_size, local_tw))
                return false;
        }
        return true;
    };

    if (inplace && fwd) {
        std::print("[Inp][Fwd][Par ]");
        std::print(perm_fmt);
        fimpl::perform_auto_size(pck_dst,
                                 pck_src,
                                 half_tw,
                                 lowk,
                                 s1_info,
                                 detail_::inplace_src,
                                 fft_size,
                                 tw,
                                 permute,
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
        std::print("[Inp][Rev][Par ]");
        std::print(perm_fmt);
        fimpl::perform_rev_auto_size(pck_dst,
                                     pck_src,
                                     half_tw,
                                     lowk,
                                     s1_info,
                                     detail_::inplace_src,
                                     fft_size,
                                     tw_rev,
                                     rev_permute,
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
        std::print("[Ext][Fwd][Par ]");
        std::print(perm_fmt);
        fimpl::perform_auto_size(pck_dst,
                                 pck_src,
                                 half_tw,
                                 lowk,
                                 s1_info,
                                 src_info,
                                 fft_size,
                                 tw,
                                 permute,
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
        std::print("[Ext][Rev][Par ]");
        std::print(perm_fmt);
        fimpl::perform_rev_auto_size(pck_dst,
                                     pck_src,
                                     half_tw,
                                     lowk,
                                     s1_info,
                                     src_info,
                                     fft_size,
                                     tw_rev,
                                     rev_permute,
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

    return true;
}

template<typename fX, uZ Width, uZ NodeSize, bool LowK, bool LocalTw, bool HalfTw>
bool seq_test_proto(meta::ce_of<permute_t> auto          perm_type,
                    const std::vector<std::complex<fX>>& signal,
                    const chk_t<fX>&                     chk_fwd,
                    const chk_t<fX>&                     chk_rev,
                    std::vector<std::complex<fX>>&       s1,
                    std::vector<fX>&                     twvec,
                    bool                                 local_check,
                    bool                                 fwd,
                    bool                                 rev,
                    bool                                 inplace,
                    bool                                 ext) {
    constexpr auto half_tw = std::bool_constant<HalfTw>{};
    constexpr auto lowk    = std::bool_constant<LowK>{};

    auto fft_size = signal.size();
    s1            = signal;

    auto& chk_fwd_ = chk_fwd(perm_type);
    auto& chk_rev_ = chk_rev(perm_type);

    using fimpl = pcx::detail_::transform<NodeSize, fX, Width>;

    auto* data_ptr = reinterpret_cast<fX*>(s1.data());

    auto tw = [&] {
        using tw_t = detail_::tw_data_t<fX, LocalTw>;
        if constexpr (LocalTw) {
            return tw_t{1, 0};
        } else {
            twvec.resize(0);
            fimpl::insert_tw_tf(twvec, fft_size, lowk, half_tw, std::true_type{});
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
    auto           perm_idxs = std::vector<u32>{};


    auto permuter = [=] {
        using enum permute_t;
        if constexpr (perm_type == bit_reversed) {
            return detail_::identity_permuter;
        } else if constexpr (perm_type == normal) {
            return detail_::br_permuter_sequential<Width, false>{};
        } else if constexpr (perm_type == shifted) {
            return detail_::br_permuter_sequential<Width, true>{};
        }
    }();
    if constexpr (perm_type != permute_t::bit_reversed) {
        using permuter_t = decltype(permuter);
        permuter         = permuter_t::insert_indexes(perm_idxs, fft_size);
        permuter.idx_ptr = perm_idxs.data();
    }
    constexpr auto perm_fmt = [=] {
        if constexpr (perm_type == permute_t::bit_reversed) {
            return "[BitRev]";
        } else if constexpr (perm_type == permute_t::normal) {
            return "[Normal]";
        } else if constexpr (perm_type == permute_t::shifted) {
            return "[Shiftd]";
        }
    }();
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
            return check_correctness(chk_fwd_, s1, Width, NodeSize, LowK, LocalTw, half_tw);
        else
            return check_correctness(chk_rev_, s1, Width, NodeSize, LowK, LocalTw, half_tw);
    };

    using dst_info_t = detail_::sequential_data_info<fX>;
    auto s1_info     = dst_info_t{{}, data_ptr};
    auto src_info = detail_::sequential_data_info<const fX>{{}, reinterpret_cast<const fX*>(signal.data())};

    if (inplace && fwd) {
        std::print("[Inp][Fwd][Seq ]");
        std::print(perm_fmt);
        s1 = signal;
        fimpl::perform_auto_size(pck_dst,
                                 pck_src,
                                 half_tw,
                                 lowk,
                                 s1_info,
                                 detail_::inplace_src,
                                 fft_size,
                                 tw,
                                 permuter);
        if (!run_check(true))
            return false;
    }
    if (inplace && rev) {
        std::print("[Inp][Rev][Seq ]");
        std::print(perm_fmt);
        s1 = signal;
        fimpl::perform_rev_auto_size(pck_dst,
                                     pck_src,
                                     half_tw,
                                     lowk,
                                     s1_info,
                                     detail_::inplace_src,
                                     fft_size,
                                     tw_rev,
                                     permuter    //
        );
        if (!run_check(false))
            return false;
    }

    if (ext && fwd) {
        std::print("[Ext][Fwd][Seq ]");
        std::print(perm_fmt);
        // s1 = signal;
        stdr::fill(s1, -69.);
        fimpl::perform_auto_size(pck_dst,
                                 pck_src,
                                 half_tw,
                                 lowk,
                                 s1_info,
                                 src_info,
                                 fft_size,
                                 tw,
                                 permuter    //
        );
        if (!run_check(true))
            return false;
    }
    if (ext && rev) {
        std::print("[Ext][Rev][Seq ]");
        std::print(perm_fmt);
        // s1 = signal;
        stdr::fill(s1, -69.);
        fimpl::perform_rev_auto_size(pck_dst,
                                     pck_src,
                                     half_tw,
                                     lowk,
                                     s1_info,
                                     src_info,
                                     fft_size,
                                     tw_rev,
                                     permuter);
        if (!run_check(false))
            return false;
    }

    s1              = signal;
    using fimpl_seq = pcx::detail_::sequential_subtransform<NodeSize, fX, Width>;
    auto coh_align  = fimpl_seq::get_align_node_seq(fft_size);
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
                    fimpl_seq::insert_tw_seq(twvec,
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
        std::print("[Inp][Fwd][Seqc]");
        std::print(perm_fmt);
        s1            = signal;
        auto tw_coh_c = tw_coh_fwd;
        fimpl_seq::perform(pck_dst,
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
        auto(permuter).sequential_permute(pck_dst, pck_dst, s1_info, detail_::inplace_src);
        if (!run_check(true))
            return false;
    }
    if (rev) {
        std::print("[Inp][Rev][Seqc]");
        std::print(perm_fmt);
        s1            = signal;
        auto tw_coh_c = tw_coh_rev;
        auto(permuter).sequential_permute(pck_src, pck_src, s1_info, detail_::inplace_src);
        fimpl_seq::perform(pck_dst,
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

template<typename fX, uZ VecWidth, uZ NodeSize, bool... LowK, bool... LocalTw, permute_t... Perm>
bool parc_run_tests(meta::val_seq<LowK...>,
                    meta::val_seq<LocalTw...>,
                    meta::val_seq<Perm...>,
                    parc_data<const fX> signal,
                    parc_data<fX>       s1,
                    uZ                  data_size,
                    const chk_t<fX>&    chk_fwd,
                    const chk_t<fX>&    chk_rev,
                    std::vector<fX>&    twvec,
                    bool                local_check,
                    bool                fwd,
                    bool                rev,
                    bool                inplace,
                    bool                external) {
    auto lk_passed = [&](auto lowk) {
        auto ltw_passed = [&](auto local_tw) {
            auto perm_passed = [&](auto perm_type) {
                return data_size <= NodeSize
                       || parc_test_proto(uZ_ce<NodeSize>{},
                                          uZ_ce<VecWidth>{},
                                          lowk,
                                          local_tw,
                                          perm_type,
                                          signal,
                                          s1,
                                          data_size,
                                          chk_fwd,
                                          chk_rev,
                                          twvec,
                                          local_check,
                                          fwd,
                                          rev,
                                          inplace,
                                          external);
            };
            return (perm_passed(val_ce<Perm>{}) && ...);
        };
        return (ltw_passed(val_ce<LocalTw>{}) && ...);
    };
    return (lk_passed(val_ce<LowK>{}) && ...);
}
template<typename fX, uZ VecWidth, uZ NodeSize, bool... LowK, bool... LocalTw, permute_t... Perm>
bool par_run_tests(meta::val_seq<LowK...>,
                   meta::val_seq<LocalTw...>,
                   meta::val_seq<Perm...>,
                   const std_vec2d<fX>& signal,
                   std_vec2d<fX>&       s1,
                   const chk_t<fX>&     chk_fwd,
                   const chk_t<fX>&     chk_rev,
                   std::vector<fX>&     twvec,
                   bool                 local_check,
                   bool                 fwd,
                   bool                 rev,
                   bool                 inplace,
                   bool                 external) {
    auto lk_passed = [&](auto lowk) {
        auto ltw_passed = [&](auto local_tw) {
            auto perm_passed = [&](auto perm_type) {
                return signal.size() <= NodeSize
                       || par_test_proto(uZ_ce<NodeSize>{},
                                         uZ_ce<VecWidth>{},
                                         lowk,
                                         local_tw,
                                         perm_type,
                                         signal,
                                         s1,
                                         chk_fwd,
                                         chk_rev,
                                         twvec,
                                         local_check,
                                         fwd,
                                         rev,
                                         inplace,
                                         external);
            };
            return (perm_passed(val_ce<Perm>{}) && ...);
        };
        return (ltw_passed(val_ce<LocalTw>{}) && ...);
    };
    return (lk_passed(val_ce<LowK>{}) && ...);
}

template<typename fX,
         uZ VecWidth,
         uZ NodeSize,
         bool... low_k,
         bool... local_tw,
         bool... half_tw,
         permute_t... Perm>
bool seq_run_tests(meta::val_seq<low_k...>,
                   meta::val_seq<local_tw...>,
                   meta::val_seq<half_tw...>,
                   meta::val_seq<Perm...>,
                   const std::vector<std::complex<fX>>& signal,
                   const chk_t<fX>&                     chk_fwd,
                   const chk_t<fX>&                     chk_rev,
                   std::vector<std::complex<fX>>&       s1,
                   std::vector<fX>&                     twvec,
                   bool                                 local_check,
                   bool                                 fwd,
                   bool                                 rev,
                   bool                                 inplace,
                   bool                                 ext) {
    auto lk_passed = [&]<bool LowK>(val_ce<LowK>) {
        auto ltw_passed = [&]<bool LocalTw>(val_ce<LocalTw>) {
            auto htw_passed = [&]<bool HalfTw>(val_ce<HalfTw>) {
                auto perm_passed = [&](auto perm_type) {
                    return signal.size() <= NodeSize * VecWidth
                           || seq_test_proto<fX, VecWidth, NodeSize, LowK, LocalTw, HalfTw>(perm_type,
                                                                                            signal,
                                                                                            chk_fwd,
                                                                                            chk_rev,
                                                                                            s1,
                                                                                            twvec,
                                                                                            local_check,
                                                                                            fwd,
                                                                                            rev,
                                                                                            inplace,
                                                                                            ext);
                };
                return (perm_passed(val_ce<Perm>{}) && ...);
            };
            return (htw_passed(val_ce<half_tw>{}) && ...);
        };
        return (ltw_passed(val_ce<local_tw>{}) && ...);
    };
    return (lk_passed(val_ce<low_k>{}) && ...);
};
}    // namespace pcx::testing
