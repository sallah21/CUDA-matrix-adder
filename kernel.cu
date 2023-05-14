#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#define N 20 // rozmiar macierzy
#define blockSize 2 // rozmiar bloku AxA
cudaError_t addWithCuda(int c[N][N], const int a[N][N], const int b[N][N], unsigned int size);

__global__ void addKernel(int c[N][N], const int a[N][N], const int b[N][N], unsigned int size)
{
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row <= N && col <= N)
    {
        c[row][col] = a[row][col] + b[row][col];
    }
}

int main()
{
    const int arraySize = N;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 ,6,7,8,9, 10, 1, 2, 3, 4, 5,2,2 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 , 60,70,80,90, 100,  1, 2, 3, 4, 5,2,2 };
    int a[arraySize][arraySize] =  {0 };
    int b[arraySize][arraySize] = { 0 };
    int c[arraySize][arraySize] = { 0 };
    int i, j;
    // wypelnianie macierzy
    for (i = 0; i < arraySize; i++) {
        for (j = 0; j < arraySize; j++) {
            a[i][j] = rand() % 65;
            b[i][j] = rand() % 65;
        }
    }
    //	wyswietlenie macierzy a
    printf("MACIERZ A : \n");
    for (i = 0; i < arraySize; i++) {
        for (j = 0; j < arraySize; j++) {
            printf("%d\t", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    //	wyswietlenie macierzy b
    printf("MACIERZ B : \n");
    for (i = 0; i < arraySize; i++) {
        for (j = 0; j < arraySize; j++) {
            printf("%d\t", b[i][j]);
        }
        printf("\n");
    }
    printf("\n");



    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    //	wyswietlenie macierzy c
    printf("MACIERZ C : \n");
    for (i = 0; i < arraySize; i++) {
        for (j = 0; j < arraySize; j++) {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int c[N][N],  const int a[N][N], const int b[N][N], unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size*size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size*size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!1");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size*size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!2");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //int defsize = 5;
    //int amountofblock = std::ceil(size / (float)defsize);
    //addKernel<<<amountofblock, defsize >> >(dev_c, dev_a, dev_b);


    printf("N: %d Block Size: %d \n", N, blockSize);
    dim3 dimBlock(blockSize, blockSize);
    printf("BLOCK: %d %d \n", dimBlock.x, dimBlock.y);

    int numBlocks = std::ceil((float)N / (float)blockSize); //ilosc bloków
    printf("size:%d numblocks : %d\n",size, numBlocks);
    dim3 dimGrid(numBlocks,numBlocks);
    printf("GRID: %d %d \n", dimGrid.x, dimGrid.y);
    addKernel<<<dimGrid, dimBlock >>>((int(*)[N])dev_c, (int(*)[N])dev_a, (int(*)[N])dev_b,4);
    
    
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size*size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! 3");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
