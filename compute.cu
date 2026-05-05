#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "vector.h"
#include "config.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__global__ void computePairwiseAccelerations(vector3 *dPos,
                                             double *dMass,
                                             vector3 *dAccels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int totalPairs = NUMENTITIES * NUMENTITIES;

    if (index >= totalPairs) {
        return;
    }

    int i = index / NUMENTITIES;
    int j = index % NUMENTITIES;

    int accelIndex = i * NUMENTITIES + j;

    if (i == j) {
        dAccels[accelIndex][0] = 0.0;
        dAccels[accelIndex][1] = 0.0;
        dAccels[accelIndex][2] = 0.0;
        return;
    }

    double distance[3];

    distance[0] = dPos[i][0] - dPos[j][0];
    distance[1] = dPos[i][1] - dPos[j][1];
    distance[2] = dPos[i][2] - dPos[j][2];

    double magnitude_sq =
        distance[0] * distance[0] +
        distance[1] * distance[1] +
        distance[2] * distance[2];

    double magnitude = sqrt(magnitude_sq);

    double accelmag = -1.0 * GRAV_CONSTANT * dMass[j] / magnitude_sq;

    dAccels[accelIndex][0] = accelmag * distance[0] / magnitude;
    dAccels[accelIndex][1] = accelmag * distance[1] / magnitude;
    dAccels[accelIndex][2] = accelmag * distance[2] / magnitude;
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

    int threadsPerBlock = 256;

    int totalPairs = NUMENTITIES * NUMENTITIES;
    int pairBlocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;

    int bodyBlocks = (NUMENTITIES + threadsPerBlock - 1) / threadsPerBlock;

    computePairwiseAccelerations<<<pairBlocks, threadsPerBlock>>>(dPos, dMass, dAccels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

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