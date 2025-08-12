#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "device_structs.hpp"

namespace kernel_utils
{
    __device__ __forceinline__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
    __device__ __forceinline__ float4 f4(float x, float y, float z, float w) { return make_float4(x, y, z, w); }
    __device__ __forceinline__ float dot3(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    __device__ __forceinline__ float3 cross3(const float3 &a, const float3 &b)
    {
        return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
    __device__ __forceinline__ float length3(const float3 &v) { return sqrtf(dot3(v, v)); }
    __device__ __forceinline__ float3 normalize3(const float3 &v)
    {
        float l = length3(v);
        return l > 0.f ? f3(v.x / l, v.y / l, v.z / l) : f3(0, 0, 0);
    }
    __device__ __forceinline__ float3 add3(const float3 &a, const float3 &b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
    __device__ __forceinline__ float3 sub3(const float3 &a, const float3 &b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
    __device__ __forceinline__ float3 mulS(const float3 &a, float s) { return f3(a.x * s, a.y * s, a.z * s); }
    __device__ __forceinline__ float3 mulV(const float3 &a, const float3 &b) { return f3(a.x * b.x, a.y * b.y, a.z * b.z); }
    __device__ __forceinline__ float3 mad3(const float3 &a, float s, const float3 &b)
    { // a + s*b
        return f3(a.x + s * b.x, a.y + s * b.y, a.z + s * b.z);
    }
    __device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
}

// ====== Ray-AABB intersection ======

__device__ bool intersectAABB(
    const float3 &ro, const float3 &rd,
    const float3 &bmin, const float3 &bmax,
    float &tmin, float &tmax);

// ====== world to uvw in [0,1]======
__device__ float3 worldToUVW(const DeviceVolume &vol, const float3 &pWorld);

__device__ __forceinline__ float sampleField(const DeviceVolume &vol, const float3 uvw);

__device__ __forceinline__ float3 sampleGradient(const DeviceVolume &vol, const float3 uvw);

__device__ __forceinline__ float4 sampleTF(const DeviceTF &tf, float value);

__device__ __forceinline__ void compositeFrontToBack(float4 sampleRGBA, float opacityScale, float4 &accum);

__global__ void volumeRendererKernel(const DeviceScene scene,
                                     uchar4 *output,
                                     int width, int height);
