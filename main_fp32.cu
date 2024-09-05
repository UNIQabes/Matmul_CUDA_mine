#include <stdio.h>
#include <mma.h>
//タイリングの階層は
//ブロックタイル(シェアードメモリの容量/2くらい z方向はベクタータイルの数分行えればそれでよい)>イテレータータイル(ブロックタイル/ワープタイル)>ワープタイル(32)>ベクタータイル(kの分割)
//これに加えて、ワープシャッフルも使っていこうと思う


#define M 2048
#define N 2048
#define K 2048

#define BLOCKTILE_MSIZE 64
#define BLOCKTILE_NSIZE 64
#define BLOCKTILE_KSIZE 32

//ここから下は固定--------------
#define WARPTILE_MSIZE 32
#define WARPTILE_NSIZE 32

#define ITERTILE_MSIZE 4
#define ITERTILE_NSIZE 8

#define SHUFFLETILE_MSIZE ITERTILE_MSIZE
#define SHUFFLETILE_NSIZE 32

#define MEMACCESSTILE_MSIZE 1
#define MEMACCESSTILE_NSIZE 32

//ABの計算を行った後、その結果をCに足し合わせる
//その時、ワープ内のスレッドがメモリ上のCの値に連続アクセスするために、
//各スレッドが計算を担当するABの要素のインデックスと、Cのロード/ストア/計算したABの値との足し合わせを担当するCの要素のインデックスは変える。
//そのため、ワープタイル内のタイリングを変更し、
//ShuffleTile(4*32)->MemAccessTile(1*32)
//とする。
//ShuffleTileはワープシャッフルでそれぞれのスレッドが持つ値を交換し、各スレッドがMemAccessTile1つにつき値1つ持つようにする。


constexpr int BLOCKTILE_MDIMNUM = M/BLOCKTILE_MSIZE;
constexpr int BLOCKTILE_NDIMNUM = N/BLOCKTILE_NSIZE;
constexpr int BLOCKTILE_KDIMNUM = K/BLOCKTILE_KSIZE;


constexpr int WARPTILE_MDIMNUM_INBT = BLOCKTILE_MSIZE/WARPTILE_MSIZE;
constexpr int WARPTILE_NDIMNUM_INBT = BLOCKTILE_NSIZE/WARPTILE_NSIZE;

constexpr int ITERTILE_MDIMNUM_INWT = WARPTILE_MSIZE/ITERTILE_MSIZE;
constexpr int ITERTILE_NDIMNUM_INWT = WARPTILE_NSIZE/ITERTILE_NSIZE;

constexpr int SHUFFLETILE_MDIMNUM_INWT = WARPTILE_MSIZE/SHUFFLETILE_MSIZE;

constexpr int MEMACCESSTILE_MDIMNUM_INST = SHUFFLETILE_MSIZE/MEMACCESSTILE_MSIZE;

constexpr int WARPNUM_INBLOCK = WARPTILE_MDIMNUM_INBT*WARPTILE_NDIMNUM_INBT;
constexpr int THREADNUM_INBLOCK = WARPNUM_INBLOCK*32;



//C=A*Bを計算する
//
__global__ void MyGemm(float *C_devPtr, float *A_devPtr,float *B_devPtr)
{
	constexpr int A_1BLOCKTILESIZE=BLOCKTILE_MSIZE*BLOCKTILE_KSIZE;
	constexpr int A_LOAD1BLOCKTILETOSHARED_ITERATIONNUM=A_1BLOCKTILESIZE/THREADNUM_INBLOCK;

	constexpr int B_1BLOCKTILESIZE=BLOCKTILE_KSIZE*BLOCKTILE_NSIZE;
	constexpr int B_LOAD1BLOCKTILETOSHARED_ITERATIONNUM=B_1BLOCKTILESIZE/THREADNUM_INBLOCK;

	const int threadLinearIdxInBlock=threadIdx.x;
	const int warpLinearIdxInBlock=threadLinearIdxInBlock/32;

	//このスレッドがワープ内で何番目のスレッドか
	const int threadLinearIdxInWarp=threadIdx.x%32;

	//このスレッドがIterationTile内のどの値の計算を行うか
	const int thread_CalNumRowIdxInIT=threadLinearIdxInWarp/ITERTILE_NSIZE;
	const int thread_CalNumColIdxInIT=threadLinearIdxInWarp%ITERTILE_NSIZE;


	//このスレッドが属するワープが担当するワープタイルのブロックタイル内のインデックス
	const int Midx_ThisWarpWarpTileInBT = warpLinearIdxInBlock / WARPTILE_NDIMNUM_INBT;
	const int NIdx_ThisWarpWarpTileInBT = warpLinearIdxInBlock % WARPTILE_NDIMNUM_INBT;

	//このスレッドが属するブロックがCの何番目のブロックを担当するか
	const int Midx_ThisBlockBlockTile=blockIdx.y;
	const int Nidx_ThisBlockBlockTile=blockIdx.x;
	
	//このスレッドが属するブロックの担当するブロックタイルの要素の、C全体の中の最小インデックス
	const int Midx_ThisBlockBlockTileStartInC=Midx_ThisBlockBlockTile*BLOCKTILE_MSIZE;
	const int Nidx_ThisBlockBlockTileStartInC=Nidx_ThisBlockBlockTile*BLOCKTILE_NSIZE;

	//このスレッドが属するワープの担当するワープタイルの要素の、C全体の中の最小インデックス
	const int Midx_ThisWarpWarpTileStartInC=Midx_ThisBlockBlockTileStartInC+Midx_ThisWarpWarpTileInBT*WARPTILE_MSIZE;
	const int Nidx_ThisWarpWarpTileStartInC=Nidx_ThisBlockBlockTileStartInC+NIdx_ThisWarpWarpTileInBT*WARPTILE_NSIZE;


	//デバイスメモリからシェアードメモリーにコピーする時にシェアードメモリに列アクセスを行うので、バンクコンフリクトが起こらないように行ごとにバンクをずらしている
	__shared__ float ATransposed_OnShared[BLOCKTILE_KSIZE][BLOCKTILE_MSIZE+1];
	__shared__ float B_OnShared[BLOCKTILE_KSIZE][BLOCKTILE_NSIZE];

	//それぞれのIterTile内の自分が担当するA*Bの値の途中計算の結果を保持する
	float thisThreadCal_ABResult[ITERTILE_MDIMNUM_INWT][ITERTILE_NDIMNUM_INWT]={0};
	/*
	if(threadLinearIdxInBlock==0)
	{
		printf("%d, %d, %d\n",Midx_ThisBlockBlockTile,Nidx_ThisBlockBlockTile,threadLinearIdxInBlock);
	}
	*/
	///このブロックの計算に使うA/Bのブロックタイルの中で、blockTile_kIdx番目のブロックタイルを読み込み、それについての計算を行う。
	for(int blockTile_kIdx=0;blockTile_kIdx<BLOCKTILE_KDIMNUM;blockTile_kIdx++)
	{
		if(threadLinearIdxInBlock==0)
		{
			//printf("(%d,%d):カーネル2\n",Midx_ThisBlockBlockTile,Nidx_ThisBlockBlockTile);
		}
		__syncthreads();
		if(threadLinearIdxInBlock==0)
		{
			printf("(%d,%d):カーネル3\n",Midx_ThisBlockBlockTile,Nidx_ThisBlockBlockTile);
		}
		//Aをブロックタイルで分割した時の(blockIdx.y, blockTile_kIdx)番目のブロックをシェアードメモリに読み込む
		const int A_LoadBlockStartRowIdx=Midx_ThisBlockBlockTileStartInC;
		const int A_LoadBlockStartColIdx=blockTile_kIdx*BLOCKTILE_KDIMNUM;
		for(int i=0;i<A_LOAD1BLOCKTILETOSHARED_ITERATIONNUM;i++)
		{
			const int loadNum_LinearIdxInBlock=i*THREADNUM_INBLOCK+threadLinearIdxInBlock;
			const int loadNum_RowIdxInBlock=loadNum_LinearIdxInBlock/BLOCKTILE_KDIMNUM;
			const int loadNum_ColIdxInBlock=loadNum_LinearIdxInBlock%BLOCKTILE_KDIMNUM;
			const int loadNum_RowIdx=A_LoadBlockStartRowIdx+loadNum_RowIdxInBlock;
			const int loadNum_ColIdx=A_LoadBlockStartColIdx+loadNum_ColIdxInBlock;

			ATransposed_OnShared[loadNum_ColIdxInBlock][loadNum_RowIdxInBlock]=A_devPtr[BLOCKTILE_KSIZE*loadNum_RowIdx+loadNum_ColIdx];
			
		}
		if(threadLinearIdxInBlock==0)
		{
			printf("カーネル2\n");
		}

		//Bをブロックタイルで分割した時の(blockTile_kIdx,blockIdx.x)番目のブロックをシェアードメモリに読み込む
		const int B_LoadBlockStartRowIdx=blockTile_kIdx*BLOCKTILE_KDIMNUM;
		const int B_LoadBlockStartColIdx=Nidx_ThisBlockBlockTileStartInC;
		for(int i=0;i<B_LOAD1BLOCKTILETOSHARED_ITERATIONNUM;i++)
		{
			const int loadNum_LinearIdxInBlock=i*THREADNUM_INBLOCK+threadLinearIdxInBlock;
			const int loadNum_RowIdxInBlock=loadNum_LinearIdxInBlock/BLOCKTILE_NDIMNUM;
			const int loadNum_ColIdxInBlock=loadNum_LinearIdxInBlock%BLOCKTILE_NDIMNUM;
			const int loadNum_RowIdx=B_LoadBlockStartRowIdx+loadNum_RowIdxInBlock;
			const int loadNum_ColIdx=B_LoadBlockStartColIdx+loadNum_ColIdxInBlock;

			B_OnShared[loadNum_RowIdxInBlock][loadNum_ColIdxInBlock]=B_devPtr[BLOCKTILE_NSIZE*loadNum_RowIdx +loadNum_ColIdx];
		}
		__syncthreads();
		

		//Cの各要素はAの行ベクトルとBの列ベクトルの内積。その2つのベクトルのblockTile_kIdx*BLOCKTILE_KSIZE+num_kIdxInBT番目の値の積を求め、和に足しこむ
		for(int num_kIdxInBT=0;num_kIdxInBT<BLOCKTILE_KDIMNUM;num_kIdxInBT++)
		{
			//[行/列]方向の0~([ITERTILE_NDIMNUM_INWT/ITERTILE_MDIMNUM_INWT]-1)番目のイテレーションタイル内の、自分の担当するCの要素の計算で使う[B/A]の値を読み込む配列。
			//レジスタにキャッシュして再利用するために宣言
			float A_RegCachedNum[ITERTILE_NDIMNUM_INWT];
			float B_RegCachedNum[ITERTILE_MDIMNUM_INWT];

			//メモリバンクをフルに使って、ワープタイル中の全ての値についての計算で必要な値全てがワープ内のどこかのレジスタに読みこまれている状態にする。

			//threadLinearIdxInWarp番目のスレッドが読み込むのは、列(M)方向で(threadLinearIdxInWarp/ITERTILE_MSIZE)番目のIterationTileの計算で、(threadLinearIdxInWarp%ITERTILE_MSIZE)*ITERTILE_NSIZE+(0~ITERTILE_NSIZE)番目のスレッドが使う値。
			//-> i * ITERTILE_MSIZE + (threadLinearIdxInWarp/ITERTILE_NSIZE_INWT)%ITERTILE_MSIZE_INWT番目のスレッドが読み込んだ値を全て受け取れば良い。
			float A_BroadCastNum=ATransposed_OnShared[num_kIdxInBT][WARPTILE_MSIZE*Midx_ThisWarpWarpTileInBT+threadLinearIdxInWarp];
			//列(M)方向の0~(ITERTILE_MDIMNUM_INWT-1)番目のイテレーションタイル内の、自分の担当するCの要素の計算で使うAの値を読み込む
			for(int MDim_IterTileIdx=0;MDim_IterTileIdx<ITERTILE_MDIMNUM_INWT;MDim_IterTileIdx++)
			{
				A_RegCachedNum[MDim_IterTileIdx]=__shfl_sync(0xffffffff,A_BroadCastNum,MDim_IterTileIdx*ITERTILE_MSIZE + threadLinearIdxInBlock/ITERTILE_NSIZE);
			}


			//threadLinearIdxInWarp番目のスレッドが読み込むのは、行(N)方向で(threadLinearIdxInWarp/ITERTILE_NSIZE)番目のIterationTileの計算で、threadLinearIdxInWarp%ITERTILE_NSIZEの値が同じになるスレッドが使う値。
			//->i * ITERTILE_NSIZE + threadLinearIdxInWarp%ITERTILE_NSIZE番目のスレッドが読み込んだ値を全て受け取れば良い。
			float B_BroadCastNum=B_OnShared[num_kIdxInBT][WARPTILE_MSIZE*Midx_ThisWarpWarpTileInBT+threadLinearIdxInWarp];
			//行(N)方向の0~(ITERTILE_NDIMNUM_INWT-1)番目のイテレーションタイル内の、自分の担当するCの要素の計算で使うBの値を読み込む
			for(int NDim_IterTileIdx=0;NDim_IterTileIdx<ITERTILE_NDIMNUM_INWT;NDim_IterTileIdx++)
			{
				B_RegCachedNum[NDim_IterTileIdx]=__shfl_sync(0xffffffff,B_BroadCastNum,NDim_IterTileIdx*ITERTILE_NSIZE + threadLinearIdxInBlock%ITERTILE_NSIZE);
			}
			
			//(MDim_IterTileIdx,NDim_IterTileIdx)番目のIterTileを内の自分が担当するCの値に足し込む積を計算する
			for(int MDim_IterTileIdx=0;MDim_IterTileIdx<ITERTILE_MDIMNUM_INWT;MDim_IterTileIdx++)
			{
				
				for(int NDim_IterTileIdx=0;NDim_IterTileIdx<ITERTILE_NDIMNUM_INWT;NDim_IterTileIdx++)
				{
					thisThreadCal_ABResult[MDim_IterTileIdx][NDim_IterTileIdx]=A_RegCachedNum[MDim_IterTileIdx]*B_RegCachedNum[NDim_IterTileIdx];
				}
			}
		}
	}
	
	//Cの値をグローバルメモリから読み込む
	//0~7番のスレッドは8~15番のスレッドに(1,0)~(1,7)の値をもらう。
	//8~15番のスレッドは16~23番のスレッドに(2,8)~(2,15)の値をもらう。
	//16~23番のスレッドは24~31番のスレッドに(3,16)~(3,23)の値をもらう。
	//24~31番のスレッドは0~7番のスレッドに(0,24)~(0,31)の値をもらう。
	//続いて
	//0~7番のスレッドは16~23番のスレッドに(2,0)~(2,7)の値をもらう。
	//8~15番のスレッドは24~31番のスレッドに(3,8)~(3,15)の値をもらう。
	//16~23番のスレッドは0~7番のスレッドに(0,16)~(0,23)の値をもらう。
	//24~31番のスレッドは8~15番のスレッドに(1,24)~(1,31)の値をもらう。

	//一般化すると
	//i=0~2
	//0~7番のスレッドは(8+8i)%32~(15+8i)%32番のスレッドに((1+i)%4,0)~((1+i)%4,7)の値をもらう。
	//8~15番のスレッドは(16+8i)%32~(23+8i)%32番のスレッドに((2+i)%4,8)~((2+i)%4,15)の値をもらう。
	//16~23番のスレッドは(24+8i)%32~(31+8i)%32番のスレッドに((3+i)%4,16)~((3+i)%4,23)の値をもらう。
	//24~31番のスレッドは(0+8i)%32~(7+8i)%32番のスレッドに(i%4,24)~(i%4,31)の値をもらう。
	
	//さらにこの繰り返しで32*32のワープタイル内の値を全て取得するため、
	//k=0~7
	//i=0~2
	//0~7番のスレッドは(8+8i)%32~(15+8i)%32番のスレッドに(k*4+(1+i)%4,0)~(k*4+(1+i)%4,7)の値をもらう。
	//8~15番のスレッドは(16+8i)%32~(23+8i)%32番のスレッドに(k*4+(2+i)%4,8)~(k*4+(2+i)%4,15)の値をもらう。
	//16~23番のスレッドは(24+8i)%32~(31+8i)%32番のスレッドに(k*4+(3+i)%4,16)~(k*4+(3+i)%4,23)の値をもらう。
	//24~31番のスレッドは(0+8i)%32~(7+8i)%32番のスレッドに(k*4+i%4,24)~(k*4+i%4,31)の値をもらう。

	//これをスレッドごとに一般化する。スレッド番号をIDとして。
	//k=0~7
	//i=0~3
	//(ID+8+8i)%32番のスレッドに(k*4+(ID/8+i)%4,ID)の値をもらう。

	//イテレーションの変数名を変えて
	//shuffleTile_MIdxInWT=0~7
	//memAccessTile_MWrapNumInST=0~3
	//(ID+8+8memAccessTile_MWrapNumInST)%32番のスレッドに(shuffleTile_MIdxInWT*4+(ID/8+memAccessTile_MWrapNumInST)%4,ID)の値をもらう。

	//このスレッドがグローバルメモリに書き込む値を格納する。
	float C_On1ShuffleTile[SHUFFLETILE_MSIZE];

	for(int shuffleTile_MIdxInWT=0;shuffleTile_MIdxInWT<SHUFFLETILE_MDIMNUM_INWT;shuffleTile_MIdxInWT++)
	{
		//他のワープからABの計算結果の値を受け取る
		for(int memAccessTile_MWrapInST=0;memAccessTile_MWrapInST<MEMACCESSTILE_MDIMNUM_INST-1;memAccessTile_MWrapInST++)
		{
			const int receiveNum_BelongMemAccessTileMIdxInST=(threadLinearIdxInWarp/ITERTILE_NSIZE+1+memAccessTile_MWrapInST)%4;
			const int sendNum_BelongIterTileNIdx=(threadLinearIdxInWarp/ITERTILE_NSIZE)+(4-1-memAccessTile_MWrapInST);
			int srcLane=(threadLinearIdxInWarp+8+8*memAccessTile_MWrapInST)%32;
			C_On1ShuffleTile[receiveNum_BelongMemAccessTileMIdxInST]=__shfl_sync(0xffffffff,srcLane,thisThreadCal_ABResult[shuffleTile_MIdxInWT][sendNum_BelongIterTileNIdx]);
		}
		//Cの元の値にABを足し合わせる
		//Cの値をグローバルメモリに書き込む
		const int MIdx_ShuffleTileStartInC=Midx_ThisWarpWarpTileStartInC+shuffleTile_MIdxInWT*SHUFFLETILE_MSIZE;
		const int NIdx_ShuffleTileStartInC=Nidx_ThisWarpWarpTileStartInC;

		for(int memAccessTile_MIdxInST=0;memAccessTile_MIdxInST<MEMACCESSTILE_MDIMNUM_INST;memAccessTile_MIdxInST++)
		{
			int MIdx_AccessInC=MIdx_ShuffleTileStartInC+memAccessTile_MIdxInST;
			int NIdx_AccessInC=NIdx_ShuffleTileStartInC+threadLinearIdxInWarp;
			C_devPtr[MIdx_AccessInC*N+NIdx_AccessInC]=C_On1ShuffleTile[memAccessTile_MIdxInST];
		}
	}

	
}



//今回はブロックタイル > イテレータータイル > ワープタイルにする
int main()
{
	cudaStream_t stream;
    cudaStreamCreate(&stream);         

	//どちらも行優先順
	float *A_hostPtr=(float *)malloc(sizeof(float)*M*K);
	float *B_hostPtr=(float *)malloc(sizeof(float)*K*N);
	float *C_hostPtr=(float *)malloc(sizeof(float)*M*N);

	float *A_devPtr;
	float *B_devPtr;
	float *C_devPtr;

	cudaMalloc(&A_devPtr,sizeof(float)*M*K);
	cudaMalloc(&B_devPtr,sizeof(float)*K*N);
	cudaMalloc(&C_devPtr,sizeof(float)*M*N);

	for(int m=0;m<M;m++)
	{
		for(int k=0;k<K;k++)
		{
			A_hostPtr[m*K+k]=1;
		}
	}
	for(int k=0;k<K;k++)
	{
		for(int n=0;n<N;n++)
		{
			B_hostPtr[k*N+n]=1;
		}
	}
	for(int m=0;m<M;m++)
	{
		for(int n=0;n<N;n++)
		{
			C_hostPtr[m*N+n]=m+n;
		}
	}

	cudaMemcpy(A_devPtr, A_hostPtr, M*K*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_devPtr, B_hostPtr, K*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_devPtr, C_hostPtr, M*N*sizeof(float), cudaMemcpyHostToDevice);

	dim3 const threadBlock_dim{BLOCKTILE_NDIMNUM, BLOCKTILE_MDIMNUM, 1};
	dim3 const thread_dim{WARPNUM_INBLOCK*32, 1, 1};
	printf("んなぁ\n");
	MyGemm<<<threadBlock_dim,thread_dim,0,stream>>>(C_devPtr,A_devPtr,B_devPtr);
	printf("んなぁ2\n");
	
	cudaMemcpy(C_hostPtr,C_devPtr,  M*N*sizeof(float), cudaMemcpyDeviceToHost);

	for(int m=0;m<10;m++)
	{
		for(int n=0;n<10;n++)
		{
			printf("%lf ",C_hostPtr[m*N+n]);
		}
		printf("\n");
	}
}