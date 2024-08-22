//fast thread index(x)をBとCの列インデックス(実際にメモリ上で配列の値が並んでいる行列の次元)を求めるのに使っている

#include <iostream>
#include <cstddef>
#include <stdio.h>

using namespace std;


template <typename T>
__global__ void gemm_v01(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        //Aの行ベクトル、Bの列ベクトルの内積を求めている
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v01(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v01<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
    //CHECK_LAST_CUDA_ERROR();
    
}

#define M 64
#define N 64
#define K 64

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

    launch_gemm_kernel_v01<float>((size_t)M,(size_t)N,(size_t)K,&alpha,MatrixA_Dev,mkMatrix_AryStride,MatrixB_Dev,knMatrix_AryStride,&beta,MatrixC_Dev,mnMatrix_AryStride,stream);

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

