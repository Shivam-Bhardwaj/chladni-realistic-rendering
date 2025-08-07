#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <stdexcept>

// RAII CUDA memory management
template<typename T>
class CudaDevicePtr {
private:
    T* ptr = nullptr;
    size_t size_bytes = 0;
    
public:
    CudaDevicePtr() = default;
    
    // Constructor with size
    explicit CudaDevicePtr(size_t count) {
        allocate(count);
    }
    
    // Move constructor
    CudaDevicePtr(CudaDevicePtr&& other) noexcept 
        : ptr(other.ptr), size_bytes(other.size_bytes) {
        other.ptr = nullptr;
        other.size_bytes = 0;
    }
    
    // Move assignment
    CudaDevicePtr& operator=(CudaDevicePtr&& other) noexcept {
        if (this != &other) {
            free();
            ptr = other.ptr;
            size_bytes = other.size_bytes;
            other.ptr = nullptr;
            other.size_bytes = 0;
        }
        return *this;
    }
    
    // Disable copy
    CudaDevicePtr(const CudaDevicePtr&) = delete;
    CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;
    
    ~CudaDevicePtr() {
        free();
    }
    
    void allocate(size_t count) {
        free(); // Free any existing allocation
        
        size_bytes = count * sizeof(T);
        cudaError_t err = cudaMalloc(&ptr, size_bytes);
        
        if (err != cudaSuccess) {
            ptr = nullptr;
            size_bytes = 0;
            throw std::runtime_error(std::string("CUDA malloc failed: ") + cudaGetErrorString(err));
        }
        
        // Initialize to zero
        err = cudaMemset(ptr, 0, size_bytes);
        if (err != cudaSuccess) {
            cudaFree(ptr);
            ptr = nullptr;
            size_bytes = 0;
            throw std::runtime_error(std::string("CUDA memset failed: ") + cudaGetErrorString(err));
        }
    }
    
    void free() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            size_bytes = 0;
        }
    }
    
    T* get() const { return ptr; }
    size_t size() const { return size_bytes; }
    bool valid() const { return ptr != nullptr; }
    
    // Convenience operators
    operator T*() const { return ptr; }
    T* operator->() const { return ptr; }
};

// RAII CUDA graphics resource management
class CudaGraphicsResource {
private:
    cudaGraphicsResource* resource = nullptr;
    bool mapped = false;
    
public:
    CudaGraphicsResource() = default;
    
    ~CudaGraphicsResource() {
        unregister();
    }
    
    // Disable copy, allow move
    CudaGraphicsResource(const CudaGraphicsResource&) = delete;
    CudaGraphicsResource& operator=(const CudaGraphicsResource&) = delete;
    
    CudaGraphicsResource(CudaGraphicsResource&& other) noexcept 
        : resource(other.resource), mapped(other.mapped) {
        other.resource = nullptr;
        other.mapped = false;
    }
    
    CudaGraphicsResource& operator=(CudaGraphicsResource&& other) noexcept {
        if (this != &other) {
            unregister();
            resource = other.resource;
            mapped = other.mapped;
            other.resource = nullptr;
            other.mapped = false;
        }
        return *this;
    }
    
    void registerBuffer(unsigned int vbo, unsigned int flags = cudaGraphicsRegisterFlagsNone) {
        unregister(); // Clean up any existing resource
        
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&resource, vbo, flags);
        if (err != cudaSuccess) {
            resource = nullptr;
            throw std::runtime_error(std::string("CUDA graphics register failed: ") + cudaGetErrorString(err));
        }
    }
    
    void* map(size_t* size = nullptr) {
        if (!resource) {
            throw std::runtime_error("Cannot map unregistered CUDA graphics resource");
        }
        
        if (mapped) {
            throw std::runtime_error("CUDA graphics resource already mapped");
        }
        
        cudaError_t err = cudaGraphicsMapResources(1, &resource, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA graphics map failed: ") + cudaGetErrorString(err));
        }
        
        void* ptr = nullptr;
        size_t bytes = 0;
        err = cudaGraphicsResourceGetMappedPointer(&ptr, &bytes, resource);
        if (err != cudaSuccess) {
            cudaGraphicsUnmapResources(1, &resource, 0);
            throw std::runtime_error(std::string("CUDA graphics get pointer failed: ") + cudaGetErrorString(err));
        }
        
        mapped = true;
        if (size) *size = bytes;
        return ptr;
    }
    
    void unmap() {
        if (mapped && resource) {
            cudaError_t err = cudaGraphicsUnmapResources(1, &resource, 0);
            mapped = false;
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA graphics unmap failed: ") + cudaGetErrorString(err));
            }
        }
    }
    
    void unregister() {
        unmap(); // Unmap first if mapped
        
        if (resource) {
            cudaGraphicsUnregisterResource(resource);
            resource = nullptr;
        }
    }
    
    bool valid() const { return resource != nullptr; }
    bool is_mapped() const { return mapped; }
};

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                               " CUDA error: " + cudaGetErrorString(err)); \
    } \
} while(0)

// Memory info utilities
struct CudaMemoryInfo {
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
    
    static CudaMemoryInfo get() {
        CudaMemoryInfo info;
        CUDA_CHECK(cudaMemGetInfo(&info.free_bytes, &info.total_bytes));
        info.used_bytes = info.total_bytes - info.free_bytes;
        return info;
    }
    
    double free_mb() const { return free_bytes / (1024.0 * 1024.0); }
    double total_mb() const { return total_bytes / (1024.0 * 1024.0); }
    double used_mb() const { return used_bytes / (1024.0 * 1024.0); }
    double usage_percent() const { return 100.0 * used_bytes / total_bytes; }
};