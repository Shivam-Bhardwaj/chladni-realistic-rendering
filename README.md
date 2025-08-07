# Chladni Plate Simulation - Realistic Rendering Engine

**v1.2.0** - A high-performance, real-time Chladni plate simulation with a professional-grade realistic rendering engine. This project transforms the visualization from basic point sprites to a publication-quality scientific visualization using modern computer graphics techniques.

## What's New in v1.2.0

### Interactive Performance Control
- **Dynamic Particle Count**: Adjust from 5,000 to 50,000 particles in real-time
- **Performance-Quality Slider**: Visual indicators for optimal settings
- **FPS Monitoring**: Color-coded performance feedback
- **Memory Diagnostics**: Live CUDA memory usage tracking

### Universal Display Support
- **Auto-Resolution Detection**: Works on any screen from 1080p to 8K+
- **Adaptive UI Scaling**: Interface scales perfectly to your display
- **Full Screen Support**: Option for immersive full-screen experience
- **Dynamic Aspect Ratio**: Maintains correct proportions on any display

### Enhanced User Experience
- **Smooth GUI Interaction**: No camera rotation while adjusting controls
- **Intelligent Force Distribution**: Particles spread naturally across the plate
- **Hot-reload Settings**: Apply changes without restarting simulation
- **Smart Performance Hints**: Get recommendations based on your hardware

## Features

### Realistic Rendering Engine
- **Physically-Based Rendering (PBR)**: Cook-Torrance BRDF with proper material properties
- **3D Particle Visualization**: Instanced billboards with volumetric appearance
- **Metallic Plate Surface**: Realistic brushed aluminum with reflections
- **Real-time Density Mapping**: GPU compute shader-based heat visualization
- **Dynamic Lighting**: Professional laboratory lighting simulation

### Scientific Accuracy
- **Authentic Chladni Physics**: Accurate vibration patterns and particle behavior
- **Material Realism**: Sand/rice particles on metal plate representation
- **Pattern Analysis**: Density visualization for studying particle accumulation
- **Scalable Particle Count**: 5K to 50K particles with performance-quality tradeoff
- **Anti-Clustering Algorithm**: Improved particle distribution prevents center-bias

### High Performance
- **CUDA Acceleration**: GPU-based physics simulation with enhanced memory management
- **Optimized Rendering**: Instanced rendering with modern OpenGL 4.5+
- **Adaptive Performance**: 20-60+ FPS depending on particle count and quality settings
- **Memory Efficient**: RAII-based CUDA memory management with live monitoring
- **Universal Scaling**: Optimized for hardware from GTX 1060 to RTX 4090+

### Performance Benchmarks
| Particle Count | Quality Level | RTX 3070 Performance | RTX 4080 Performance |
|----------------|---------------|----------------------|----------------------|
| 5,000         | High Performance | 60+ FPS             | 60+ FPS              |
| 15,000        | Balanced        | 45-60 FPS           | 60+ FPS              |
| 30,000        | High Quality    | 30-45 FPS           | 50-60 FPS            |
| 50,000        | Maximum Detail  | 20-35 FPS           | 35-50 FPS            |

## Mathematical Foundation

The simulation implements the Chladni plate equation with multiple vibration modes:

$$z(x, y, t) = A \sum_{m,n} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right) \cos(\omega t + \phi_{mn})$$

Where:
- $z(x, y, t)$ is the vertical displacement at position $(x, y)$ and time $t$
- $A$ is the amplitude scaling factor
- $L$ is the plate dimension
- $\omega = 2\pi f$ is the angular frequency
- $\phi_{mn}$ are mode-specific phase shifts

The particle dynamics follow:

$$m\frac{d^2\vec{r}}{dt^2} = \vec{F}_{plate} + \vec{F}_{damping} + \vec{F}_{gravity} + \vec{F}_{collision}$$

Where the plate force is derived from the gradient:
$$\vec{F}_{plate} = -m\nabla z(x, y, t)$$

## Quick Start

### Prerequisites
- **NVIDIA GPU** with CUDA support (GTX 1060+ recommended, RTX series preferred)
- **Visual Studio 2019+** (Windows) or **GCC 9+** (Linux)
- **CMake 3.18+**
- **CUDA Toolkit 11.0+** (12.0+ recommended for latest optimizations)

### Dependencies (via vcpkg)
```bash
vcpkg install glew glfw3 glm imgui portaudio fftw3
```

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/Shivam-Bhardwaj/chladni-realistic-rendering.git
cd chladni-realistic-rendering

# Create build directory
mkdir build && cd build

# Configure with CMake (with optimizations)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build with maximum performance
cmake --build . --config Release --parallel

# Run the simulation
./ChladniSimulation.exe  # Windows
./ChladniSimulation     # Linux
```

### Audio Setup Tips
```bash
# Test your microphone before running:
# Windows: Check "Sound" in Control Panel
# Linux: Use 'arecord -l' to list audio devices
# Make sure microphone permissions are enabled
```

### Troubleshooting

**Performance Issues:**
- Lower particle count via GUI slider
- Enable "High Performance" mode in GPU settings
- Close other GPU-intensive applications
- Check CUDA memory usage in diagnostics panel

**Display Issues:**
- Update NVIDIA drivers to latest version
- Try windowed mode if fullscreen has problems
- Adjust UI scaling if interface appears too small/large

**Audio Issues:**
- Switch to "Manual Control" mode in GUI
- Check microphone permissions in system settings
- Try different audio devices in system audio settings
- Increase audio sensitivity slider if patterns don't respond

## Controls & User Interface

### Mouse Controls
- **Left Click + Drag**: Rotate camera around the Chladni plate
- **Mouse Wheel**: Zoom in/out for detailed pattern observation
- **Right Click**: Reset camera to default position
- **Ctrl + Scroll**: Manual frequency control (100-2000 Hz)
- **Shift + Scroll**: Manual amplitude adjustment (0.0-0.2)

### Audio Input
- **Live Microphone**: Speak, whistle, or play music to see real-time patterns
- **Frequency Detection**: Different tones create different geometric patterns
- **Amplitude Response**: Louder sounds create more particle movement
- **Audio Sensitivity**: Adjustable via GUI for optimal responsiveness

### GUI Control Panel (Press 'G')
- **Audio Mode Toggle**: Switch between live input and manual control
- **Particle Count Slider**: Adjust 5K-50K particles with performance indicators
- **Frequency Presets**: Quick buttons for common frequencies (100Hz, 440Hz, 800Hz, etc.)
- **Amplitude Presets**: Low (0.02), Medium (0.05), High (0.1) settings
- **Performance Monitor**: Live FPS, memory usage, and optimization suggestions
- **Diagnostics Panel**: CUDA status, audio status, and debug information

### Interactive Features
- **Hot-reload Particles**: Apply new particle counts instantly
- **Performance Warnings**: Visual alerts when FPS drops below optimal
- **Smart Recommendations**: Automatic suggestions for better performance
- **Memory Tracking**: Real-time CUDA memory usage with optimization tips

### Keyboard Shortcuts
- **G**: Toggle GUI control panel
- **ESC**: Exit simulation
- **SPACE**: Toggle rendering modes (if using comparison demo)
- **F**: Cycle through frequency presets in manual mode

## Technical Architecture

### Rendering Pipeline
- **OpenGL 4.5+**: Modern graphics with compute shader support
- **Physically-Based Rendering**: Cook-Torrance BRDF implementation
- **Instanced Particle Rendering**: Efficient GPU-based particle visualization
- **Dynamic Resolution Scaling**: Automatic adaptation to display resolution
- **Real-time Density Mapping**: GPU compute shaders for pattern analysis

### Physics Simulation
- **CUDA Acceleration**: Massively parallel particle physics on GPU
- **Enhanced Particle Distribution**: Circular initialization prevents clustering
- **Center-damping Forces**: Distance-based scaling eliminates center-bias
- **Adaptive Force Scaling**: Maintains stability across different particle counts
- **Robust Boundary Conditions**: Intelligent particle containment and reset

### Memory Management
- **RAII-based CUDA Resources**: Automatic cleanup and error recovery
- **Dynamic Buffer Allocation**: Efficient memory scaling with particle count
- **Live Memory Monitoring**: Real-time usage tracking and optimization
- **Resource Pooling**: Optimized GPU memory utilization

### Audio Processing
- **Real-time FFT**: High-performance frequency analysis with FFTW3
- **PortAudio Integration**: Cross-platform audio input handling
- **Adaptive Audio Sensitivity**: Dynamic range adjustment
- **Frequency-to-Pattern Mapping**: Scientifically accurate audio visualization

## System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GTX 1060 / RTX 2060 (CUDA Compute 6.1+)
- **RAM**: 8GB system memory
- **VRAM**: 4GB GPU memory
- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600
- **Resolution**: 1080p display

### Recommended for Optimal Experience
- **GPU**: NVIDIA RTX 3070 / RTX 4070 or better
- **RAM**: 16GB+ system memory
- **VRAM**: 8GB+ GPU memory
- **CPU**: Intel i7-10700K / AMD Ryzen 7 3700X or better
- **Resolution**: 1440p+ display

### Maximum Quality Settings
- **GPU**: NVIDIA RTX 4080 / RTX 4090
- **RAM**: 32GB system memory
- **VRAM**: 12GB+ GPU memory
- **Display**: 4K/5K/8K high-resolution display
- **Audio**: Professional microphone for best pattern response

## Use Cases

### Scientific Research
- **Pattern Analysis**: Study Chladni plate vibration modes
- **Educational Demonstrations**: Interactive physics visualization
- **Frequency Response Studies**: Analyze material resonance characteristics
- **Publication-Quality Visuals**: Professional scientific visualization

### Graphics Development
- **PBR Rendering Reference**: Modern physically-based rendering implementation
- **CUDA Integration Example**: GPU physics and OpenGL interoperability
- **Performance Optimization**: Scaling techniques for real-time applications
- **UI/UX Design**: Adaptive interface scaling and user experience

### Entertainment & Art
- **Interactive Audio Visualization**: Music-responsive particle systems
- **Digital Art Creation**: Generate unique geometric patterns
- **Live Performance Tool**: Real-time audio-visual experiences
- **Educational Gaming**: Physics-based interactive learning

*Transforming scientific simulation visualization with cutting-edge graphics techniques and adaptive performance optimization.*
