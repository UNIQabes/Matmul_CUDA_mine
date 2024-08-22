//ワープまで2次元にする必要ある?
//fast thread index(スレッドのxyz中で、同じワープのスレッドが並んでいる次元)がxであるのに、AとCについて、それを列のインデックス(行列の次元中で、実際に隣接する配列の値が並んでいる次元)ではなく行のインデックスを求めるのに使ってしまった。
//AとCが同じワープのスレッドでメモリアクセスされる時、非連続アクセスになってしまっている。(幅[lda/ldc]*sizeof(T)のストライドアクセスになっている)

template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
		//Aの行ベクトルとBの列ベクトルの内積を求めている。
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
			//ldaとldbはおそらくA,Bのストライド(行配列の先頭の間隔)
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};//grid_dim{m/block_dim.x,n/block_dim.y}だが、割り算で切り上げて欲しいからこの形に
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}