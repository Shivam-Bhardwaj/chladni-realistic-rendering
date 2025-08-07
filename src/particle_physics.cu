#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <vector_functions.h>
#include <ctime>
#include <cmath>
#include "debug_info.h"

// CUDA helper functions for float3 operations
__device__ __host__ inline float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __host__ inline float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __host__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __host__ inline float3 normalize(float3 v) {
    float len = length(v);
    if (len > 1e-12f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

struct Particle {
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
};

struct SimulationParams {
    float frequency;
    float amplitude;
    float damping;
    float plateSize;
    float gravity;
    float restitution;
    float friction;
    float dt;
    int numParticles;
};

// Device-side debug counters
struct DeviceDebugInfo {
    unsigned int nan_position_count;
    unsigned int nan_velocity_count;
    unsigned int inf_value_count;
    unsigned int invalid_normal_count;
    unsigned int oob_count;
    unsigned int last_bad_index;
};

__device__ DeviceDebugInfo d_debug;

__device__ inline void recordInvalidNormal(int idx) {
    atomicAdd(&d_debug.invalid_normal_count, 1u);
    d_debug.last_bad_index = static_cast<unsigned int>(idx);
}

__device__ inline void recordNaNPos(int idx) {
    atomicAdd(&d_debug.nan_position_count, 1u);
    d_debug.last_bad_index = static_cast<unsigned int>(idx);
}

__device__ inline void recordNaNVel(int idx) {
    atomicAdd(&d_debug.nan_velocity_count, 1u);
    d_debug.last_bad_index = static_cast<unsigned int>(idx);
}

__device__ inline void recordInf(int idx) {
    atomicAdd(&d_debug.inf_value_count, 1u);
    d_debug.last_bad_index = static_cast<unsigned int>(idx);
}

__device__ float3 calculatePlateVibration(float3 pos, float time, SimulationParams params) {
    // Enhanced Chladni plate equation with multiple modes for better patterns
    float halfPlate = params.plateSize * 0.5f;
    
    // Normalize position to [-1, 1]
    float nx = pos.x / halfPlate;
    float ny = pos.y / halfPlate;
    
    // Multiple frequency modes for richer patterns - adjusted for larger plate
    float baseFreq = params.frequency * 0.008f; // Slightly lower scale for larger plate patterns
    
    // Primary mode
    float kx1 = baseFreq * CUDART_PI_F;
    float ky1 = baseFreq * CUDART_PI_F;
    
    // Secondary mode for complexity
    float kx2 = baseFreq * CUDART_PI_F * 1.41f; // sqrt(2) ratio
    float ky2 = baseFreq * CUDART_PI_F * 1.73f; // sqrt(3) ratio
    
    float omega = 2.0f * CUDART_PI_F * params.frequency * 0.1f; // Slower oscillation
    
    // Combine multiple modes
    float z1 = sinf(kx1 * nx) * sinf(ky1 * ny);
    float z2 = sinf(kx2 * nx) * cosf(ky2 * ny) * 0.5f; // Secondary mode with lower amplitude
    
    float z = params.amplitude * (z1 + z2) * cosf(omega * time);
    
    // Calculate spatial gradients (forces that push particles)
    float dz_dx = params.amplitude * cosf(omega * time) * (
        kx1 * cosf(kx1 * nx) * sinf(ky1 * ny) / halfPlate +
        kx2 * cosf(kx2 * nx) * cosf(ky2 * ny) * 0.5f / halfPlate
    );
    
    float dz_dy = params.amplitude * cosf(omega * time) * (
        ky1 * sinf(kx1 * nx) * cosf(ky1 * ny) / halfPlate -
        ky2 * sinf(kx2 * nx) * sinf(ky2 * ny) * 0.5f / halfPlate
    );
    
    // Time derivative for vertical acceleration
    float dz_dt = -params.amplitude * omega * (z1 + z2) * sinf(omega * time);
    
    return make_float3(dz_dx, dz_dy, z + dz_dt * 0.1f); // Scale vertical motion
}

__global__ void updateParticles(Particle* particles, SimulationParams params, float time, curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.numParticles) return; // Don't count this as OOB - it's just thread overflow
    
    Particle& p = particles[idx];
    curandState localRandState = randStates[idx];
    
    // Reset force
    p.force = make_float3(0.0f, 0.0f, -params.gravity * p.mass);
    
    // Get plate vibration at particle position
    float3 plateEffect = calculatePlateVibration(p.position, time, params);
    
    // Apply Chladni forces - particles are pushed away from vibration antinodes
    // This creates the characteristic patterns where particles collect at nodes (quiet areas)
    
    // Horizontal forces from plate gradient (main Chladni effect)
    // Reduce force near center to prevent clustering
    float distanceFromCenter = sqrtf(p.position.x * p.position.x + p.position.y * p.position.y);
    float centerDamping = fmaxf(0.3f, fminf(1.0f, distanceFromCenter / (params.plateSize * 0.2f)));
    
    float forceScale = p.mass * 18.0f * centerDamping; // Reduced near center
    p.force.x += plateEffect.x * forceScale;
    p.force.y += plateEffect.y * forceScale;
    
    // Check if particle is on or near the plate
    float plateHeight = plateEffect.z;
    if (p.position.z <= plateHeight + 0.02f) {
        // Vertical collision with vibrating plate
        float3 normal_raw = make_float3(-plateEffect.x * 0.1f, -plateEffect.y * 0.1f, 1.0f);
        float3 normal = normalize(normal_raw);
        if (normal.x == 0.0f && normal.y == 0.0f && normal.z == 0.0f) { recordInvalidNormal(idx); }
        
        float3 relativeVel = p.velocity;
        float normalVel = dot(relativeVel, normal);
        
        // Bounce off vibrating plate
        if (normalVel < 0 || p.position.z < plateHeight) {
            // Collision response with energy from plate vibration
            float bounceForce = (-normalVel * (1.0f + params.restitution) + fabsf(plateEffect.z) * 10.0f) / params.dt;
            p.force.z += bounceForce * p.mass;
            
            // Horizontal friction reduces sliding
            float3 tangentVel = relativeVel - normal * normalVel;
            float tangentSpeed = length(tangentVel);
            if (tangentSpeed > 0.001f) {
                float3 frictionForce = -normalize(tangentVel) * params.friction * bounceForce * 0.5f;
                p.force.x += frictionForce.x;
                p.force.y += frictionForce.y;
            }
        }
        
        // Keep particle slightly above plate
        if (p.position.z < plateHeight + 0.001f) {
            p.position.z = plateHeight + 0.001f;
        }
    }
    
    // Add tiny random perturbation for realism (much reduced)
    p.force.x += (curand_uniform(&localRandState) - 0.5f) * 0.0005f;
    p.force.y += (curand_uniform(&localRandState) - 0.5f) * 0.0005f;
    
    // Apply stronger damping to prevent runaway particles
    float dampingFactor = params.damping * 2.0f;  // Stronger damping
    p.force.x -= dampingFactor * p.velocity.x;
    p.force.y -= dampingFactor * p.velocity.y;
    p.force.z -= dampingFactor * p.velocity.z;
    
    // Update velocity and position (Verlet integration)
    float3 acceleration = make_float3(p.force.x / p.mass, p.force.y / p.mass, p.force.z / p.mass);
    p.velocity.x += acceleration.x * params.dt;
    p.velocity.y += acceleration.y * params.dt;
    p.velocity.z += acceleration.z * params.dt;
    
    // Limit velocity to avoid numerical blow-ups and escaping particles
    const float maxSpeed = 2.0f;  // Much lower max speed
    float speed = length(p.velocity);
    if (speed > maxSpeed) {
        float scale = maxSpeed / speed;
        p.velocity.x *= scale;
        p.velocity.y *= scale;
        p.velocity.z *= scale;
    }
    
    p.position.x += p.velocity.x * params.dt;
    p.position.y += p.velocity.y * params.dt;
    p.position.z += p.velocity.z * params.dt;
    
    // Detect invalid numbers and clamp positions
    if (!isfinite(p.position.x) || !isfinite(p.position.y) || !isfinite(p.position.z)) {
        recordNaNPos(idx);
        if (!isfinite(p.position.x)) p.position.x = 0.0f;
        if (!isfinite(p.position.y)) p.position.y = 0.0f;
        if (!isfinite(p.position.z)) p.position.z = 0.01f;
    }
    if (!isfinite(p.velocity.x) || !isfinite(p.velocity.y) || !isfinite(p.velocity.z)) {
        recordNaNVel(idx);
        if (!isfinite(p.velocity.x)) p.velocity.x = 0.0f;
        if (!isfinite(p.velocity.y)) p.velocity.y = 0.0f;
        if (!isfinite(p.velocity.z)) p.velocity.z = 0.0f;
    }
    
    // Robust boundary enforcement with particle reset for extreme cases
    float halfPlate = params.plateSize * 0.4f;  // Safe boundary
    bool isOutOfBounds = false;
    
    // Check if particle is severely out of bounds (reset case)
    if (fabsf(p.position.x) > params.plateSize || 
        fabsf(p.position.y) > params.plateSize || 
        p.position.z > 1.0f || p.position.z < -0.1f) {
        
        atomicAdd(&d_debug.oob_count, 1u);
        isOutOfBounds = true;
        
        // Reset particle to safe position
        p.position.x = (curand_uniform(&localRandState) - 0.5f) * halfPlate;
        p.position.y = (curand_uniform(&localRandState) - 0.5f) * halfPlate;
        p.position.z = 0.02f + curand_uniform(&localRandState) * 0.03f;
        p.velocity = make_float3(0.0f, 0.0f, 0.0f);
        p.force = make_float3(0.0f, 0.0f, -params.gravity * p.mass);
    }
    
    if (!isOutOfBounds) {
        // Normal boundary enforcement with soft walls
        float margin = 0.05f; // Soft boundary margin
        
        // X boundaries with progressive force
        if (p.position.x > halfPlate - margin) {
            float overshoot = p.position.x - (halfPlate - margin);
            p.force.x -= overshoot * 1000.0f * p.mass; // Strong restoring force
            if (p.position.x > halfPlate) {
                p.position.x = halfPlate;
                p.velocity.x = -fabsf(p.velocity.x) * 0.2f; // Always push inward
            }
        } else if (p.position.x < -(halfPlate - margin)) {
            float overshoot = -(halfPlate - margin) - p.position.x;
            p.force.x += overshoot * 1000.0f * p.mass;
            if (p.position.x < -halfPlate) {
                p.position.x = -halfPlate;
                p.velocity.x = fabsf(p.velocity.x) * 0.2f; // Always push inward
            }
        }
        
        // Y boundaries with progressive force
        if (p.position.y > halfPlate - margin) {
            float overshoot = p.position.y - (halfPlate - margin);
            p.force.y -= overshoot * 1000.0f * p.mass;
            if (p.position.y > halfPlate) {
                p.position.y = halfPlate;
                p.velocity.y = -fabsf(p.velocity.y) * 0.2f;
            }
        } else if (p.position.y < -(halfPlate - margin)) {
            float overshoot = -(halfPlate - margin) - p.position.y;
            p.force.y += overshoot * 1000.0f * p.mass;
            if (p.position.y < -halfPlate) {
                p.position.y = -halfPlate;
                p.velocity.y = fabsf(p.velocity.y) * 0.2f;
            }
        }
        
        // Z boundaries
        if (p.position.z < 0.0f) {
            p.position.z = 0.001f;
            p.velocity.z = fabsf(p.velocity.z) * 0.3f; // Bounce upward
        } else if (p.position.z > 0.3f) {
            float overshoot = p.position.z - 0.3f;
            p.force.z -= overshoot * 2000.0f * p.mass; // Strong downward force
            if (p.position.z > 0.5f) {
                p.position.z = 0.5f;
                p.velocity.z = -fabsf(p.velocity.z) * 0.5f;
            }
        }
    }
    
    // Save random state
    randStates[idx] = localRandState;
}

__global__ void initRandomStates(curandState* states, unsigned long seed, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void initParticlePositions(Particle* particles, SimulationParams params, curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.numParticles) return;
    
    curandState localRandState = randStates[idx];
    
    // Initialize particles in a more distributed pattern to avoid center clustering
    float halfPlate = params.plateSize * 0.4f;
    
    // Use different distribution patterns to avoid center bias
    float rand1 = curand_uniform(&localRandState);
    float rand2 = curand_uniform(&localRandState);
    
    // Create spiral/circular distribution to better spread particles
    float angle = rand1 * 2.0f * CUDART_PI_F;
    float radius = sqrtf(rand2) * halfPlate; // sqrt for uniform area distribution
    
    particles[idx].position.x = radius * cosf(angle);
    particles[idx].position.y = radius * sinf(angle);
    particles[idx].position.z = 0.01f + curand_uniform(&localRandState) * 0.03f;
    
    // Initialize with zero velocity
    particles[idx].velocity = make_float3(0.0f, 0.0f, 0.0f);
    particles[idx].force = make_float3(0.0f, 0.0f, 0.0f);
    particles[idx].mass = 0.001f;
    
    // Save random state
    randStates[idx] = localRandState;
}

extern "C" {
    void initParticles(Particle* d_particles, SimulationParams params, curandState* d_randStates) {
        int blockSize = 512;  // Larger block size for better GPU utilization with more particles
        int numBlocks = (params.numParticles + blockSize - 1) / blockSize;
        
        // Reset debug counters
        resetDebugInfo();
        
        // First initialize random states
        initRandomStates<<<numBlocks, blockSize>>>(d_randStates, time(NULL), params.numParticles);
        cudaDeviceSynchronize();  // Wait for random states to be ready
        
        // Then initialize particle positions using the random states
        initParticlePositions<<<numBlocks, blockSize>>>(d_particles, params, d_randStates);
        cudaDeviceSynchronize();  // Wait for particle initialization to complete
    }
    
    void stepSimulation(Particle* d_particles, SimulationParams params, float time, curandState* d_randStates) {
        int blockSize = 512;  // Match initialization block size
        int numBlocks = (params.numParticles + blockSize - 1) / blockSize;
        updateParticles<<<numBlocks, blockSize>>>(d_particles, params, time, d_randStates);
    }

    void resetDebugInfo() {
        DeviceDebugInfo zero = {};
        cudaMemcpyToSymbol(d_debug, &zero, sizeof(DeviceDebugInfo));
    }

    void fetchDebugInfo(DebugInfo* hostOut) {
        DeviceDebugInfo tmp;
        cudaMemcpyFromSymbol(&tmp, d_debug, sizeof(DeviceDebugInfo));
        hostOut->nan_position_count = tmp.nan_position_count;
        hostOut->nan_velocity_count = tmp.nan_velocity_count;
        hostOut->inf_value_count = tmp.inf_value_count;
        hostOut->invalid_normal_count = tmp.invalid_normal_count;
        hostOut->oob_count = tmp.oob_count;
        hostOut->last_bad_index = tmp.last_bad_index;
    }
}