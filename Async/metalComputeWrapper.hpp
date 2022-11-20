//
//  metalComputeWrapper.hpp
//  Async
//
//  Created by Likor7 on 14.11.2022.
//

#ifndef metalComputeWrapper_hpp
#define metalComputeWrapper_hpp

#include <Metal/Metal.hpp>

class metalComputeWrapper {
public:
    MTL::Device* mDevice;
    
    // The compute pipeline generated from the compute kernel in the .metal shader file.
    MTL::ComputePipelineState* mComputeFunctionPSO;
    
    // The command queue used to pass commands to the device.
    MTL::CommandQueue* mCommandQueue;
    
    //Data
    int rows1;
    int columns1;
    float *arrayMatrix1;
    int rows2;
    int columns2;
    float *arrayMatrix2;
    
    // Buffers to hold data.
    MTL::Buffer *mBufferA;
    MTL::Buffer *mBufferB;
    MTL::Buffer *mBufferResult;
    MTL::Buffer *mBufferCol1;
    
    void init(MTL::Device*, int rows1_, int columns1_, float *arrayMatrix1_, int rows2_, int columns2_, float *arrayMatrix2_);
    void prepareData();
    void sendComputeCommand();
    
private:
    void encodeComputeCommand(MTL::ComputeCommandEncoder*);
    void generateBuffer(MTL::Buffer*,  float* arrayMatrix, int rows, int columns);
    void verifyResults();
};

#endif /* metalComputeWrapper_hpp */
