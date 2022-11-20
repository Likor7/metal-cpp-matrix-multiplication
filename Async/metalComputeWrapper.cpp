//
//  metalComputeWrapper.cpp
//  Async
//
//  Created by Likor7 on 14.11.2022.
//

#include "metalComputeWrapper.hpp"
#include <iostream>

void metalComputeWrapper::init(MTL::Device* device, int rows1_, int columns1_, float *arrayMatrix1_, int rows2_, int columns2_, float *arrayMatrix2_) {
    rows1 = rows1_;
    columns1 = columns1_;
    arrayMatrix1 = arrayMatrix1_;
    
    rows2 = rows2_;
    columns2 = columns2_;
    arrayMatrix2 = arrayMatrix2_;
    
    mDevice = device;
    NS::Error* error;
    
    auto defaultLibrary = mDevice->newDefaultLibrary();
    
    if (!defaultLibrary) {
        std::cerr << "Failed to find the default library.\n";
        exit(-1);
    }
    
    auto functionName = NS::String::string("work_on_arrays", NS::ASCIIStringEncoding);
    auto computeFunction = defaultLibrary->newFunction(functionName);
    
    if(!computeFunction){
        std::cerr << "Failed to find the compute function.\n";
    }
    
    mComputeFunctionPSO = mDevice->newComputePipelineState(computeFunction, &error);
    
    if (!computeFunction) {
        std::cerr << "Failed to create the pipeline state object.\n";
        exit(-1);
    }
    
    mCommandQueue = mDevice->newCommandQueue();
    
    if (!mCommandQueue) {
        std::cerr << "Failed to find command queue.\n";
        exit(-1);
    }
}

void metalComputeWrapper::prepareData() {
    // Allocate three buffers to hold our initial data and the result.
    const unsigned int BUFFER_SIZE1 = rows1*columns1*sizeof(float);
    const unsigned int BUFFER_SIZE2 = rows2*columns2*sizeof(float);
    const unsigned int BUFFER_SIZE3 = rows1*columns2*sizeof(float);
    const unsigned int BUFFER_SIZE4 = sizeof(int);
    
    mBufferA = mDevice->newBuffer(BUFFER_SIZE1, MTL::ResourceStorageModeShared);
    mBufferB = mDevice->newBuffer(BUFFER_SIZE2, MTL::ResourceStorageModeShared);
    mBufferResult = mDevice->newBuffer(BUFFER_SIZE3, MTL::ResourceStorageModeShared);
    mBufferCol1 = mDevice->newBuffer(BUFFER_SIZE4, MTL::ResourceStorageModeShared);
    
    generateBuffer(mBufferA, arrayMatrix1, rows1, columns1);
    generateBuffer(mBufferB, arrayMatrix2, rows2, columns2);
    
    float* dataRes = (float*) mBufferResult->contents();
    
    for(int i = 0; i < rows1*columns2; i++)
    {
        dataRes[i] = 0;
    }
    
    uint* m1Ptr = (uint*) mBufferCol1->contents();
    
    m1Ptr[0] = columns1;
}

void metalComputeWrapper::generateBuffer(MTL::Buffer * buffer, float* arrayMatrix, int rows, int columns) {
    float* dataPtr = (float*) buffer->contents();
    
    for(int i = 0; i < rows*columns; i++)
        dataPtr[i] = arrayMatrix[i];
}

void metalComputeWrapper::sendComputeCommand() {
    // Create a command buffer to hold commands.
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    
    // Start a compute pass.
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);
    
    encodeComputeCommand(computeEncoder);
    
    // End the compute pass.
    computeEncoder->endEncoding();
    
    // Execute the command.
    commandBuffer->commit();
    
    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();
    
    verifyResults();
}

void metalComputeWrapper::encodeComputeCommand(MTL::ComputeCommandEncoder * computeEncoder) {
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(mComputeFunctionPSO);
    computeEncoder->setBuffer(mBufferA, 0, 0);
    computeEncoder->setBuffer(mBufferB, 0, 1);
    computeEncoder->setBuffer(mBufferResult, 0, 2);
    computeEncoder->setBuffer(mBufferCol1, 0, 3);
    
    MTL::Size gridSize = MTL::Size(rows1, columns2, 1);

    MTL::Size threadgroupSize = MTL::Size(mComputeFunctionPSO->threadExecutionWidth(),  mComputeFunctionPSO->maxTotalThreadsPerThreadgroup() / mComputeFunctionPSO->threadExecutionWidth(), 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}

void metalComputeWrapper::verifyResults(){
    float* result = (float*) mBufferResult->contents();
    std::cout<<std::endl;
    std::cout<<std::endl;
    for (int i = 0; i < rows1; i++)
    {
        for (int j = 0; j < columns2; j++)
        {
            std::cout<<result[i + rows1 * j]<<"\t";
        }
        std::cout<<std::endl;
    }

    std::cout << "Compute results as expected.\n";
}
