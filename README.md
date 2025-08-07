# Chladni Plate Simulation - Realistic Rendering Engine

A high-performance, real-time Chladni plate simulation with a professional-grade realistic rendering engine. This project transforms the visualization from basic point sprites to a publication-quality scientific visualization using modern computer graphics techniques.

## 🌟 Features

### 🎨 Realistic Rendering Engine
- **Physically-Based Rendering (PBR)**: Cook-Torrance BRDF with proper material properties
- **3D Particle Visualization**: Instanced billboards with volumetric appearance
- **Metallic Plate Surface**: Realistic brushed aluminum with reflections
- **Real-time Density Mapping**: GPU compute shader-based heat visualization
- **Dynamic Lighting**: Professional laboratory lighting simulation

### 🔬 Scientific Accuracy
- **Authentic Chladni Physics**: Accurate vibration patterns and particle behavior
- **Material Realism**: Sand/rice particles on metal plate representation
- **Pattern Analysis**: Density visualization for studying particle accumulation
- **High Particle Count**: 30,000+ particles for detailed pattern formation

### ⚡ High Performance
- **CUDA Acceleration**: GPU-based physics simulation
- **Optimized Rendering**: Instanced rendering with modern OpenGL
- **Real-time Processing**: 60+ FPS on modern hardware
- **Memory Efficient**: RAII-based CUDA memory management

## 🚀 Quick Start

### Prerequisites
- **NVIDIA GPU** with CUDA support (GTX 1060+ recommended)
- **Visual Studio 2019+** (Windows) or **GCC 9+** (Linux)
- **CMake 3.18+**
- **CUDA Toolkit 11.0+**

### Dependencies (via vcpkg)
```bash
vcpkg install glew glfw3 glm imgui portaudio fftw3
```

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/chladni-realistic-rendering.git
cd chladni-realistic-rendering

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build (Windows)
cmake --build . --config Release
```

## 🎮 Controls

### Main Simulation
- **Mouse Controls**: Left click + drag to rotate camera, scroll to zoom
- **Audio Input**: Speak/whistle near microphone to see patterns
- **Manual Mode**: Use GUI controls (G key) for precise parameter adjustment

### Comparison Demo
- **SPACE**: Toggle between basic and realistic rendering
- **A**: Enable auto-switching every 5 seconds
- **1-4**: Adjust material properties in real-time

## 📊 Technical Details

- **Rendering Engine**: Modern OpenGL 4.5+ with compute shaders
- **Physics**: CUDA-accelerated particle simulation
- **Memory Management**: RAII-based CUDA resource handling
- **Audio Processing**: Real-time FFT with PortAudio

*Transforming scientific simulation visualization with modern graphics techniques.*
