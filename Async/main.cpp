//
//  main.cpp
//  Async
//
//  Created by Likor7 on 14.11.2022.
//


#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include "metalComputeWrapper.hpp"
#include <iostream>
#include <chrono>
#include <random>

float** randomMatrix(int rows, int columns){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(1, 10);
    
    float **matrix = new float *[rows];
    for(int i = 0; i < rows; i++)
    {
        matrix[i] = new float[columns];
        for(int j = 0; j < columns; j++)
        {
            matrix[i][j] = dist(gen);
        }
    }
    return matrix;
}

void printMatrix(float** matrix, int rows, int columns)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < columns; j++)
        {
            std::cout<<matrix[i][j]<<"\t";
        }
        std::cout<<std::endl;
    }
}

float* convertMatrixToArray(float** matrix, int rows, int columns)
{
    float* arrayMatrix = new float[rows*columns];
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            arrayMatrix[i + rows * j] = matrix[i][j];
        }
    }
    
    return arrayMatrix;
}

void printArrayToMatrix(float *arrayMatrix, int rows, int columns)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            std::cout<<arrayMatrix[i + rows * j]<<"\t";
        }
        std::cout<<std::endl;
    }
}

float** initMatrix(int rows, int columns)
{
    float **matrix = new float *[rows];
    for(int i = 0; i < rows; i++)
    {
        matrix[i] = new float[columns];
        for(int j = 0; j < columns; j++)
        {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}

void multiplyMatrix(float** matrix1, int rows1, int columns1, float** matrix2, int rows2, int columns2, float** resultMatrix)
{
    for (int i = 0; i < rows1; i++)
    {
        for (int j = 0; j < columns2; j++)
        {
            for (int k = 0; k < columns1; k++)
            {
                resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}



int main(int argc, const char * argv[]) {
    
    //TEST DATA
//    const int n1 = 3, m1 = 3;
//    const int n2 = 3, m2 = 3;
//    float data1[n1][m1] = {
//        4,9,4,
//        3,7,5,
//        9,9,7,
//    };
//    float data2[n2][m2] = {
//        6,7,6,
//        4,1,6,
//        3,7,5,
//    };
//
//    float** matrix1 = new float*[n1];
//    float** matrix2  = new float*[n2];
//
//    for(int i = 0; i < n1; i++)
//    {
//        matrix1[i] = new float[m1];
//        for(int j = 0; j < m1; j++)
//        {
//            matrix1[i][j] = data1[i][j];
//        }
//    }
//
//    for(int i = 0; i < n2; i++)
//    {
//        matrix2[i] = new float[m2];
//        for(int j = 0; j < m2; j++)
//        {
//            matrix2[i][j] = data2[i][j];
//        }
//    }
//
    
    int n1, m1, n2, m2;
    std::cout<<"Enter rows 1:\t";
    std::cin>>n1;
    std::cout<<"Enter columns 1:\t";
    std::cin>>m1;
    std::cout<<std::endl;
    std::cout<<"Enter rows 2:\t";
    std::cin>>n2;
    std::cout<<"Enter columns 2:\t";
    std::cin>>m2;

    float** matrix1 = randomMatrix(n1,m1);
    float** matrix2 = randomMatrix(n2,m2);
    
    float** result1 = initMatrix(n1, m2);
    
    auto startCPU = std::chrono::steady_clock::now();
    
    multiplyMatrix(matrix1, n1, m1, matrix2, n2, m2, result1);
    
    auto endCPU = std::chrono::steady_clock::now();
    auto CPU_time = endCPU - startCPU;
    std::cout << "Computation CPU time: "
           << std::chrono::duration <double, std::milli> (CPU_time).count()
           << " ms."
           <<std::endl;
    
//            printMatrix(matrix1, n1, m1);
//            std::cout<<std::endl<<std::endl;
//            std::cout<<std::endl<<std::endl;
//
//            printMatrix(matrix2, n2, m2);
//            std::cout<<std::endl<<std::endl;

    printMatrix(result1, n1, m2);
    std::cout<<std::endl;
    std::cout<<std::endl;

    float *arrayMatrix1 = convertMatrixToArray(matrix1, n1, m1);
    float *arrayMatrix2 = convertMatrixToArray(matrix2, n2, m2);


    NS::AutoreleasePool* pPool   = NS::AutoreleasePool::alloc()->init();

    MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();

    // Create the custom object used to encapsulate the Metal code.
    // Initializes objects to communicate with the GPU.
    metalComputeWrapper* computer = new metalComputeWrapper();

    computer->init(pDevice, n1, m1, arrayMatrix1, n2, m2, arrayMatrix2);

    // Create buffers to hold data
    computer->prepareData();

    // Time the compute phase.
    auto startGPU = std::chrono::steady_clock::now();

    // Send a command to the GPU to perform the calculation.
    computer->sendComputeCommand();

    // End of compute phase.
    auto endGPU = std::chrono::steady_clock::now();
    auto GPU_time = endGPU - startGPU;

    pPool->release();

    std::cout << "Computation GPU time: "
           << std::chrono::duration <double, std::milli> (GPU_time).count()
           << " ms."
           <<std::endl;
    
    std::cout << "\nAcceleration (S_p = CPU/GPU): = " << (double) (CPU_time).count()/(double) (GPU_time).count()  << std::endl;
     
}
