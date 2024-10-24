#include "pcx/include/simd/common.hpp"
#include "pcx/include/simd/math.hpp"
using namespace pcx;

auto simd_add(const f32* lhs, const f32* rhs, f32* out) {
    auto lhsv = simd::cxload<16>(lhs);
    auto rhsv = simd::cxload<16>(rhs);
    auto res  = simd::add(lhsv, rhsv);
    simd::cxstore<16>(out, res);
}


int main() {
    return 0;
}
