#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "vector.h"
#include "config.h"

#define TILE_SIZE 16

// Simple CUDA error checker.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)



__global__ void computePairwiseAccelerationsTiled(vector3 *dPos,
                                                  double *dMass,
                                                  vector3 *dAccels) {
    int localCol = threadIdx.x;
    int localRow = threadIdx.y;

    int j = blockIdx.x * TILE_SIZE + localCol; // source body
    int i = blockIdx.y * TILE_SIZE + localRow; // target body

    __shared__ double sourcePos[TILE_SIZE][3];
    __shared__ double targetPos[TILE_SIZE][3];
    __shared__ double sourceMass[TILE_SIZE];

    // Load source body data into shared memory.
    // One row of threads loads the source bodies for this tile.
    if (localRow == 0 && j < NUMENTITIES) {
        sourcePos[localCol][0] = dPos[j][0];
        sourcePos[localCol][1] = dPos[j][1];
        sourcePos[localCol][2] = dPos[j][2];
        sourceMass[localCol] = dMass[j];
    }

    // Load target body data into shared memory.
    // One column of threads loads the target bodies for this tile.
    if (localCol == 0 && i < NUMENTITIES) {
        targetPos[localRow][0] = dPos[i][0];
        targetPos[localRow][1] = dPos[i][1];
        targetPos[localRow][2] = dPos[i][2];
    }

    __syncthreads();

    if (i >= NUMENTITIES || j >= NUMENTITIES) {
        return;
    }

    int accelIndex = i * NUMENTITIES + j;

    if (i == j) {
        dAccels[accelIndex][0] = 0.0;
        dAccels[accelIndex][1] = 0.0;
        dAccels[accelIndex][2] = 0.0;
        return;
    }

    double dx = targetPos[localRow][0] - sourcePos[localCol][0];
    double dy = targetPos[localRow][1] - sourcePos[localCol][1];
    double dz = targetPos[localRow][2] - sourcePos[localCol][2];

    double magnitude_sq = dx * dx + dy * dy + dz * dz;

    // Safety check against divide-by-zero if two bodies ever occupy the same position.
    if (magnitude_sq == 0.0) {
        dAccels[accelIndex][0] = 0.0;
        dAccels[accelIndex][1] = 0.0;
        dAccels[accelIndex][2] = 0.0;
        return;
    }

    double magnitude = sqrt(magnitude_sq);
    double accelmag = -1.0 * GRAV_CONSTANT * sourceMass[localCol] / magnitude_sq;

    dAccels[accelIndex][0] = accelmag * dx / magnitude;
    dAccels[accelIndex][1] = accelmag * dy / magnitude;
    dAccels[accelIndex][2] = accelmag * dz / magnitude;
}

__global__ void updateBodies(vector3 *dPos,
                             vector3 *dVel,
                             vector3 *dAccels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUMENTITIES) {
        return;
    }

    double accelSum[3] = {0.0, 0.0, 0.0};

    for (int j = 0; j < NUMENTITIES; j++) {
        int accelIndex = i * NUMENTITIES + j;

        accelSum[0] += dAccels[accelIndex][0];
        accelSum[1] += dAccels[accelIndex][1];
        accelSum[2] += dAccels[accelIndex][2];
    }

    dVel[i][0] += accelSum[0] * INTERVAL;
    dVel[i][1] += accelSum[1] * INTERVAL;
    dVel[i][2] += accelSum[2] * INTERVAL;

    dPos[i][0] += dVel[i][0] * INTERVAL;
    dPos[i][1] += dVel[i][1] * INTERVAL;
    dPos[i][2] += dVel[i][2] * INTERVAL;
}

extern "C" void compute() {
    vector3 *dPos = NULL;
    vector3 *dVel = NULL;
    vector3 *dAccels = NULL;
    double *dMass = NULL;

    size_t vectorArraySize = sizeof(vector3) * NUMENTITIES;
    size_t massArraySize = sizeof(double) * NUMENTITIES;
    size_t accelMatrixSize = sizeof(vector3) * NUMENTITIES * NUMENTITIES;

    CUDA_CHECK(cudaMalloc((void **)&dPos, vectorArraySize));
    CUDA_CHECK(cudaMalloc((void **)&dVel, vectorArraySize));
    CUDA_CHECK(cudaMalloc((void **)&dMass, massArraySize));
    CUDA_CHECK(cudaMalloc((void **)&dAccels, accelMatrixSize));

    CUDA_CHECK(cudaMemcpy(dPos, hPos, vectorArraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel, vectorArraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMass, mass, massArraySize, cudaMemcpyHostToDevice));

    dim3 pairThreads(TILE_SIZE, TILE_SIZE);
    dim3 pairBlocks((NUMENTITIES + TILE_SIZE - 1) / TILE_SIZE,
                    (NUMENTITIES + TILE_SIZE - 1) / TILE_SIZE);

    computePairwiseAccelerationsTiled<<<pairBlocks, pairThreads>>>(dPos, dMass, dAccels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int threadsPerBlock = 256;
    int bodyBlocks = (NUMENTITIES + threadsPerBlock - 1) / threadsPerBlock;

    updateBodies<<<bodyBlocks, threadsPerBlock>>>(dPos, dVel, dAccels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hPos, dPos, vectorArraySize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVel, dVel, vectorArraySize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFree(dMass));
    CUDA_CHECK(cudaFree(dAccels));
}