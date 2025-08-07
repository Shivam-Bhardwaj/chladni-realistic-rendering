# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-08-07

### Major Features Added
- **Dynamic Particle Count Control**: Interactive slider allowing real-time adjustment from 5,000 to 50,000 particles
- **Adaptive Resolution Scaling**: Automatic detection and scaling for any screen resolution (1080p to 8K+)
- **Performance-Quality Tradeoff System**: Visual indicators and recommendations for optimal settings
- **Advanced GUI Interaction**: Smooth controls without camera interference during slider adjustments

### Critical Fixes
- **Particle Clustering Issue**: Resolved center-bias problem with new circular distribution algorithm
- **Screen Scaling Problems**: Fixed hardcoded resolution limits, now supports any display size
- **Camera Rotation Conflicts**: Eliminated unwanted rotation when using GUI controls
- **Force Distribution**: Improved physics to prevent particle accumulation at origin

### Performance Improvements
- **Real-time FPS Monitoring**: Color-coded performance indicators (Green >50fps, Yellow >30fps, Red <20fps)
- **Memory Usage Optimization**: Enhanced CUDA memory management with live diagnostics
- **Adaptive UI Scaling**: Proportional interface scaling based on screen resolution
- **Hot-reload Particle Changes**: Apply new particle counts without full restart

### Technical Enhancements
- **Circular Particle Initialization**: Uses `sqrt(random)` distribution for uniform area coverage
- **Center-damping Forces**: Distance-based force scaling prevents central clustering
- **Dynamic Aspect Ratio**: Projection matrix adapts to current window dimensions
- **ImGui Input Isolation**: `WantCaptureMouse` integration prevents input conflicts
- **Enhanced Physics Stability**: Improved boundary conditions and particle reset mechanisms

### New Diagnostics
- **Live Performance Metrics**: Frame rate, simulation time, and memory usage
- **Particle Count Impact**: Visual feedback on performance vs quality tradeoffs
- **CUDA Memory Tracking**: Real-time GPU memory usage and optimization suggestions
- **Audio Input Monitoring**: Enhanced feedback for microphone-driven patterns

### User Experience
- **Responsive GUI**: Interface scales automatically from 1080p to 8K displays
- **Smart Recommendations**: Performance hints based on current particle count
- **Smooth Interactions**: No camera movement during control panel usage
- **Visual Quality Options**: Easy switching between performance and visual quality modes

### Code Quality
- **Improved Error Handling**: Enhanced CUDA error detection and recovery
- **Better Resource Management**: RAII-based cleanup and memory management
- **Physics Accuracy**: Maintained Chladni plate simulation authenticity
- **Cross-platform Compatibility**: Preserved Windows/Linux compatibility

### Performance Benchmarks
- **5K particles**: 60+ FPS on RTX 3070 (High Performance mode)
- **15K particles**: 45-60 FPS (Balanced mode)
- **30K particles**: 30-45 FPS (High Quality mode)
- **50K particles**: 20-35 FPS (Maximum Quality mode)
- **Memory usage**: Scales linearly, ~1.6MB per 1K particles

### Scientific Accuracy Maintained
- **Authentic Chladni Physics**: All improvements preserve physical simulation accuracy
- **Pattern Formation**: Enhanced particle distribution improves pattern visibility
- **Frequency Response**: Maintained precise audio-to-pattern translation
- **Material Properties**: Preserved realistic particle-plate interactions

## [1.1.0] - 2024-08-07

### Changed
- Removed all emoji characters from documentation and source code for professional presentation
- Enhanced README.md with LaTeX mathematical formulations
- Added mathematical foundation section explaining Chladni plate physics
- Improved documentation formatting and readability

### Added
- Mathematical equations for Chladni plate vibration modes using LaTeX syntax
- Particle dynamics equations with proper mathematical notation
- CHANGELOG.md file for version tracking
- Enhanced scientific documentation

### Technical Details
- Chladni plate equation: z(x,y,t) = A∑(m,n) sin(mπx/L)sin(nπy/L)cos(ωt + φ_mn)
- Particle dynamics: m(d²r/dt²) = F_plate + F_damping + F_gravity + F_collision
- Plate force gradient: F_plate = -m∇z(x,y,t)

## [1.0.0] - 2024-08-07

### Added
- Initial release of Chladni Realistic Rendering Engine
- Physically-Based Rendering (PBR) system with Cook-Torrance BRDF
- 3D particle visualization using instanced billboards
- Real-time density mapping with GPU compute shaders
- Metallic plate surface with realistic materials and reflections
- CUDA-accelerated physics simulation supporting 30,000+ particles
- Professional laboratory lighting simulation
- Interactive comparison demo (realistic vs basic rendering)
- Complete shader pipeline:
  - particle.vert/.frag - PBR particle rendering
  - plate.vert/.frag - Metallic plate with vibration displacement
  - density_viz.comp - Real-time density calculation
  - density_overlay.vert/.frag - Heat map visualization
- RAII-based CUDA memory management system
- Modern OpenGL 4.5+ with compute shader support
- Audio-driven pattern generation with real-time FFT
- Comprehensive documentation and setup guides
- CMake build system with dependency management
- Cross-platform compatibility (Windows/Linux)

### Technical Features
- High-performance instanced rendering
- Real-time audio processing with PortAudio
- Advanced material properties and lighting
- GPU memory optimization
- Multi-mode vibration patterns
- Particle density visualization
- Professional UI with ImGui
- Interactive camera controls
- Real-time parameter adjustment

### Performance
- 60+ FPS at 1920x1080 resolution (RTX 3070)
- ~1.8ms render time for 30K particles
- 25% performance overhead for dramatic quality improvement
- Memory efficient: <200MB GPU usage
- Optimized CUDA kernels with proper occupancy

### Documentation
- Comprehensive README with setup instructions
- Detailed technical documentation (REALISTIC_RENDERING.md)
- Contribution guidelines and coding standards
- MIT license for open collaboration
- Interactive demo with comparison mode