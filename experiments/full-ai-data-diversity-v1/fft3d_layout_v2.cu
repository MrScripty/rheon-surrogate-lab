// Generic double-precision DCT layout operations. No physical or learned values.
extern "C" __global__ void dct_pack_even(const double* input, double* output,
                                         int nz, int ny, int nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 8 * nz * ny * nx;
    if (i >= total) return;
    int x = i % (2 * nx), y = (i / (2 * nx)) % (2 * ny);
    int z = i / (4 * nx * ny);
    x = x < nx ? x : 2 * nx - 1 - x;
    y = y < ny ? y : 2 * ny - 1 - y;
    z = z < nz ? z : 2 * nz - 1 - z;
    output[i] = input[(z * ny + y) * nx + x];
}

extern "C" __global__ void dct_unpack_forward(const double2* spectrum,
        const double* rotation_real, const double* rotation_imag, double* output,
        int nz, int ny, int nx, int stride_z, int stride_y, int stride_x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz * ny * nx) return;
    int x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    double2 value = spectrum[z * stride_z + y * stride_y + x * stride_x];
    output[i] = value.x * rotation_real[i] - value.y * rotation_imag[i];
}

extern "C" __global__ void dct_pack_inverse(const double* coefficients,
        const double* rotation_real, const double* rotation_imag, double2* spectrum,
        int nz, int ny, int nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 4 * nz * ny * (nx + 1);
    if (i >= total) return;
    int x = i % (nx + 1), y = (i / (nx + 1)) % (2 * ny);
    int z = i / ((nx + 1) * (2 * ny));
    if (x == nx || y == ny || z == nz) {
        spectrum[i] = make_double2(0., 0.);
        return;
    }
    int az = z < nz ? z : 2 * nz - z;
    int ay = y < ny ? y : 2 * ny - y;
    double value = coefficients[(az * ny + ay) * nx + x];
    spectrum[i] = make_double2(value * rotation_real[i], value * rotation_imag[i]);
}

extern "C" __global__ void dct_unpack_inverse(const double* input, double* output,
                                              int nz, int ny, int nx, int stride_z, int stride_y, int stride_x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz * ny * nx) return;
    int x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    output[i] = input[z * stride_z + y * stride_y + x * stride_x];
}
