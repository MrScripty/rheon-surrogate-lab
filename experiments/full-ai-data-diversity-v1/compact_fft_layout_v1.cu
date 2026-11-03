// Generic orthonormal DCT from length-N real FFTs, no physical coefficients.
__device__ __forceinline__ int length(int axis, int nz, int ny, int nx) {
    return axis == 0 ? nz : (axis == 1 ? ny : nx);
}
__device__ __forceinline__ void position(int batch, int q, int axis,
        int nz, int ny, int nx, int& z, int& y, int& x) {
    if (axis == 0) { z = q; y = batch / nx; x = batch % nx; }
    else if (axis == 1) { z = batch / nx; y = q; x = batch % nx; }
    else { z = batch / ny; y = batch % ny; x = q; }
}
__device__ __forceinline__ int batch_at(int z, int y, int x, int axis,
                                       int nz, int ny, int nx) {
    return axis == 0 ? y * nx + x : (axis == 1 ? z * nx + x : z * ny + y);
}
__device__ __forceinline__ int coordinate(int z, int y, int x, int axis) {
    return axis == 0 ? z : (axis == 1 ? y : x);
}
__device__ __forceinline__ int permute(int j, int n) {
    return j < (n + 1) / 2 ? 2 * j : 2 * (n - 1 - j) + 1;
}
__device__ __forceinline__ int unpermute(int q, int n) {
    return q % 2 == 0 ? q / 2 : n - 1 - q / 2;
}
__device__ __forceinline__ double read_intermediate(const double* input,
        int z, int y, int x, int previous, int nz, int ny, int nx, int s0, int s1) {
    if (previous < 0) return input[(z * ny + y) * nx + x];
    int q = coordinate(z, y, x, previous);
    int batch = batch_at(z, y, x, previous, nz, ny, nx);
    return input[batch * s0 + unpermute(q, length(previous, nz, ny, nx)) * s1];
}

extern "C" __global__ void compact_forward_pack(const double* input, double* output,
        int nz, int ny, int nx, int axis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz * ny * nx) return;
    int n = length(axis, nz, ny, nx), z, y, x;
    position(i / n, permute(i % n, n), axis, nz, ny, nx, z, y, x);
    output[i] = input[(z * ny + y) * nx + x];
}

extern "C" __global__ void compact_forward_transition(const double2* input,
        const double* phase_real, const double* phase_imag,
        const double* divisor, double* output, int nz, int ny, int nx,
        int previous, int next, int s0, int s1, int apply_response) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz * ny * nx) return;
    int z, y, x;
    if (next < 0) { z = i / (nx * ny); y = (i / nx) % ny; x = i % nx; }
    else {
        int n = length(next, nz, ny, nx);
        position(i / n, permute(i % n, n), next, nz, ny, nx, z, y, x);
    }
    int n = length(previous, nz, ny, nx);
    int k = coordinate(z, y, x, previous);
    int batch = batch_at(z, y, x, previous, nz, ny, nx);
    bool reflected = k > n / 2;
    double2 v = input[batch * s0 + (reflected ? n - k : k) * s1];
    double imag = reflected ? -v.y : v.y;
    double value = v.x * phase_real[k] + imag * phase_imag[k];
    if (apply_response) value = i == 0 ? 0. : value / divisor[i];
    output[i] = value;
}

extern "C" __global__ void compact_inverse_transition(const double* input,
        const double* cosine, const double* sine, const double* inverse_alpha,
        void* output, int nz, int ny, int nx, int previous, int next,
        int s0, int s1, int gauge) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (next < 0) {
        if (i >= nz * ny * nx) return;
        int z = i / (nx * ny), y = (i / nx) % ny, x = i % nx;
        double value = read_intermediate(input, z, y, x, previous, nz, ny, nx, s0, s1);
        static_cast<double*>(output)[i] = gauge ? value - input[0] : value;
        return;
    }
    int n = length(next, nz, ny, nx), f = n / 2 + 1;
    int total = (nz * ny * nx / n) * f;
    if (i >= total) return;
    int k = i % f, batch = i / f, z, y, x;
    position(batch, k, next, nz, ny, nx, z, y, x);
    double uk = read_intermediate(input, z, y, x, previous, nz, ny, nx, s0, s1) * inverse_alpha[k];
    if (k == 0) { static_cast<double2*>(output)[i] = make_double2(uk, 0.); return; }
    position(batch, n - k, next, nz, ny, nx, z, y, x);
    double un = read_intermediate(input, z, y, x, previous, nz, ny, nx, s0, s1) * inverse_alpha[n - k];
    double real = uk * cosine[k] + un * sine[k];
    double imag = (n % 2 == 0 && k == n / 2) ? 0. : uk * sine[k] - un * cosine[k];
    static_cast<double2*>(output)[i] = make_double2(real, imag);
}
