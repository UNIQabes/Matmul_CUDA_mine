#include <iostream>
#include <cstddef>
#include <stdio.h>

using namespace std;


template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(T const* A, size_t lda,
                                           T const* B, size_t ldb,
                                           T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                                           T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n,
                                           size_t k)
{
    // thread_block_tile_idx:何番目のタイルを取ってくるか(Aの(m,k)、Bの(k,n)番目のタイルを取ってくる)
    // Load data from A on DRAM to A_thread_block_tile on shared memory.
//CUDAにおけるunrollってどういう意図で行うものなの?TensorCoreを使ってくれるとか?
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
             NUM_THREADS;
         ++load_idx)
    {

        //シェアードメモリ上の配列のインデックス
        //thread_linear_idx=0~NUM_THREADS-1
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            BLOCK_TILE_SIZE_K};
        //グローバル配列上の配列のインデックス
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};

        //行列の要素を行優先でNUM_THREADS個読み込む
        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.

        //static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==0U);

        // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
        //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        // {
        //     A_thread_block_tile[A_thread_block_tile_row_idx]
        //                        [A_thread_block_tile_col_idx] = val;
        // }
        A_thread_block_tile[A_thread_block_tile_row_idx]
                           [A_thread_block_tile_col_idx] = val;
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
             NUM_THREADS;
         ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            BLOCK_TILE_SIZE_X};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.

        //static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==0U);
		
        // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
        //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        // {
        //     B_thread_block_tile[B_thread_block_tile_row_idx]
        //                        [B_thread_block_tile_col_idx] = val;
        // }
        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] = val;
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    //ブロックに属するスレッドの総数 = thread_linear_idxの最大値+1
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    //ブロック内のスレッド全てを1列に並べた時、何個目か
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    //Yはm Xはnの次元
    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    //A(m,k)とB(k,n)のタイルをシェアードメモリにコピー -> タイルの値同士をかけて部分行列積を計算する　　をkのループで繰り返す。 
    //ここでは、kがthread_block_tile_idxに置き換わっている    
    T sum{static_cast<T>(0)};
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        //シェアードメモリのポインタもグローバル領域のメモリと同じ型として扱える
        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                   BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        //syncthreadsってどの単位での同期?同じブロック内?
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // Doing this results in 2 TOPS.
            // Suppose blockDim.x = blockDim.y = 32.
            // Effectively, for a warp, in one iteration, we read the value from
            // A_thread_block_tile at the same location on the shared memory
            // resulting in a broadcast, we also read 32 values that have no
            // bank conflicts from B_thread_block_tile. Even with that, all the
            // values have to be read from the shared memory and consequence is
            // the shared memory instruction runs very intensively just to
            // compute a small number of values using simple arithmetic
            // instructions, which is not efficient.
            sum += A_thread_block_tile[threadIdx.y][k_i] *
                   B_thread_block_tile[k_i][threadIdx.x];
        }
        __syncthreads();
    }
    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    //static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    //static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v02<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    //CHECK_LAST_CUDA_ERROR();
}



#define M 256
#define N 256
#define K 256

int main()
{
    cout << "ulololol" << endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);         

    size_t mkMatrix_AryStride=K+2;
    size_t mkMatrix_ArySize=M*mkMatrix_AryStride;
    size_t knMatrix_AryStride=N+2;
    size_t knMatrix_ArySize=K*knMatrix_AryStride;
    size_t mnMatrix_AryStride=N+2;
    size_t mnMatrix_ArySize=M*mnMatrix_AryStride;

    float *MatrixA_Host=(float *)malloc(mkMatrix_ArySize*sizeof(float));
    float *MatrixB_Host=(float *)malloc(knMatrix_ArySize*sizeof(float));
    float *MatrixC_Host=(float *)malloc(knMatrix_ArySize*sizeof(float));
    

    float *MatrixA_Dev, *MatrixB_Dev, *MatrixC_Dev;
    cudaMalloc(&MatrixA_Dev,mkMatrix_ArySize*sizeof(float));
    cudaMalloc(&MatrixB_Dev,knMatrix_ArySize*sizeof(float));
    cudaMalloc(&MatrixC_Dev,mnMatrix_ArySize*sizeof(float));

    
    for(int m=0;m<M;m++)
    {
        for(int k=0;k<K;k++)
        {
            MatrixA_Host[m*mkMatrix_AryStride+k]=1;
        }
    }
    for(int k=0;k<K;k++)
    {
        for(int n=0;n<N;n++)
        {
            MatrixB_Host[k*knMatrix_AryStride+n]=1;
        }
    }
    for(int m=0;m<M;m++)
    {
        for(int n=0;n<N;n++)
        {
            MatrixC_Host[m*mnMatrix_AryStride+n]=m+n;
        }
    }

    cudaMemcpy(MatrixA_Dev, MatrixA_Host, mkMatrix_ArySize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(MatrixB_Dev, MatrixB_Host, knMatrix_ArySize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(MatrixC_Dev, MatrixC_Host, mnMatrix_ArySize*sizeof(float), cudaMemcpyHostToDevice);

    float alpha=1;
    float beta=1;

    launch_gemm_kernel_v02<float>((size_t)M,(size_t)N,(size_t)K,&alpha,MatrixA_Dev,mkMatrix_AryStride,MatrixB_Dev,knMatrix_AryStride,&beta,MatrixC_Dev,mnMatrix_AryStride,stream);

    cudaMemcpy(MatrixA_Host, MatrixA_Dev, mkMatrix_ArySize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(MatrixB_Host, MatrixB_Dev, knMatrix_ArySize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(MatrixC_Host, MatrixC_Dev, mnMatrix_ArySize*sizeof(float), cudaMemcpyDeviceToHost);

    for(int m=0;m<5;m++)
    {
        for(int n=0;n<5;n++)
        {
            cout << MatrixC_Host[m*mnMatrix_AryStride+n] << " ";
        }
        cout << endl;
    }

    //launch_gemm_kernel_v01(size_t m, size_t n, size_t k, T const* alpha,T const* A, size_t lda, T const* B, size_t ldb,T const* beta, T* C, size_t ldc,cudaStream_t stream)
                            

}
