// Fused execution of the existing fixed convex neural field readout.
// Inputs are four float32 carrier fields and neural displacement tensors.
// No velocity backtrace, pressure solve, reference state or fitted coefficient.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" int rheon_neural_field_readout(
    const float* fields, const void* shifts, float* output,
    std::int64_t nz, std::int64_t ny, std::int64_t nx, int shift_kind) {
    if (!fields || !shifts || !output || nz < 1 || ny < 1 || nx < 1 ||
        nz > 2048 || ny > 2048 || nx > 2048 || (shift_kind != 0 && shift_kind != 1)) return 1;
    const std::int64_t size = nz * ny * nx;
    for (std::int64_t i = 0; i < 4 * size; ++i) {
        if (!std::isfinite(fields[i])) return 2;
    }
    const auto shift_at = [&](std::int64_t index) -> double {
        return shift_kind == 0 ? static_cast<double>(static_cast<const float*>(shifts)[index])
                               : static_cast<const double*>(shifts)[index];
    };
    for (std::int64_t i = 0; i < 12 * size; ++i) {
        const double shift = shift_at(i);
        if (!std::isfinite(shift)) return 3;
        if (std::abs(shift) > 1.0) return 4;
    }
    for (std::int64_t field = 0; field < 4; ++field) {
        const std::int64_t field_start = field * size;
        for (std::int64_t z = 0; z < nz; ++z) {
            for (std::int64_t y = 0; y < ny; ++y) {
                for (std::int64_t x = 0; x < nx; ++x) {
                    const std::int64_t index = x + nx * (y + ny * z);
                    const double px = std::clamp(static_cast<double>(x) + shift_at((field * 3) * size + index), 0.0, static_cast<double>(nx - 1));
                    const double py = std::clamp(static_cast<double>(y) + shift_at((field * 3 + 1) * size + index), 0.0, static_cast<double>(ny - 1));
                    const double pz = std::clamp(static_cast<double>(z) + shift_at((field * 3 + 2) * size + index), 0.0, static_cast<double>(nz - 1));
                    const auto lx = static_cast<std::int64_t>(std::floor(px));
                    const auto ly = static_cast<std::int64_t>(std::floor(py));
                    const auto lz = static_cast<std::int64_t>(std::floor(pz));
                    const auto hx = std::min(lx + 1, nx - 1);
                    const auto hy = std::min(ly + 1, ny - 1);
                    const auto hz = std::min(lz + 1, nz - 1);
                    const double fx = px - static_cast<double>(lx);
                    const double fy = py - static_cast<double>(ly);
                    const double fz = pz - static_cast<double>(lz);
                    double value = 0.0;
                    double lower = std::numeric_limits<double>::infinity();
                    double upper = -std::numeric_limits<double>::infinity();
                    // Match the reference z/y/x donor order and x/y/z products.
                    for (int bz = 0; bz < 2; ++bz) {
                        for (int by = 0; by < 2; ++by) {
                            for (int bx = 0; bx < 2; ++bx) {
                                const auto ix = bx ? hx : lx;
                                const auto iy = by ? hy : ly;
                                const auto iz = bz ? hz : lz;
                                const double sample = static_cast<double>(fields[field_start + ix + nx * (iy + ny * iz)]);
                                double weight = 1.0;
                                weight = weight * (bx ? fx : 1.0 - fx);
                                weight = weight * (by ? fy : 1.0 - fy);
                                weight = weight * (bz ? fz : 1.0 - fz);
                                value = value + weight * sample;
                                lower = std::min(lower, sample);
                                upper = std::max(upper, sample);
                            }
                        }
                    }
                    output[field_start + index] = static_cast<float>(std::min(std::max(value, lower), upper));
                }
            }
        }
    }
    return 0;
}
