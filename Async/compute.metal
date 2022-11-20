//
//  compute.metal
//  Async
//
//  Created by Likor7 on 14.11.2022.
//

#include <metal_stdlib>
using namespace metal;


kernel void work_on_arrays(device const float* inA  [[ buffer(0) ]],
                           device const float* inB [[ buffer(1) ]],
                           device float* result [[ buffer(2) ]],
                           device const uint* m1 [[ buffer(3) ]],
                           uint2 thread_id [[thread_position_in_threadgroup]],
                           uint2 block_dim [[threads_per_threadgroup]],
                           uint2 block_id [[threadgroup_position_in_grid]] )
{
    uint row = block_id.y * block_dim.y + thread_id.y;
    uint col = block_id.x * block_dim.x + thread_id.x;
    
    uint n = m1[0];
    
    if( col < n && row < n) {
        for(uint i = 0; i < n; i++) {
            result[row + n * col] += inA[n * i + row] * inB[i + n * col];
        }
    }
}
