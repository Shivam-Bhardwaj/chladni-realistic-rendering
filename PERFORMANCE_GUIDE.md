# Performance Optimization Guide

## Getting Maximum Performance

### Hardware-Specific Optimizations

#### NVIDIA RTX 40-Series (RTX 4060-4090)
- **Optimal Particle Count**: 35,000-50,000
- **Expected Performance**: 45-60+ FPS at 4K
- **Memory Usage**: 8-12GB VRAM recommended
- **Special Features**: RT cores enhance PBR rendering quality

#### NVIDIA RTX 30-Series (RTX 3060-3090)
- **Optimal Particle Count**: 25,000-40,000
- **Expected Performance**: 30-50 FPS at 4K, 60+ FPS at 1440p
- **Memory Usage**: 6-10GB VRAM recommended
- **Best Settings**: Balanced mode with 30K particles

#### NVIDIA GTX 16-Series / RTX 20-Series
- **Optimal Particle Count**: 15,000-25,000
- **Expected Performance**: 30-45 FPS at 1440p, 45+ FPS at 1080p
- **Memory Usage**: 4-6GB VRAM recommended
- **Best Settings**: High Performance mode with 20K particles

#### Older Hardware (GTX 1060, GTX 1070, GTX 1080)
- **Optimal Particle Count**: 5,000-15,000
- **Expected Performance**: 30+ FPS at 1080p
- **Memory Usage**: 3-4GB VRAM recommended
- **Best Settings**: High Performance mode with 10K particles

### Performance Tuning Steps

1. **Start with Auto-Detection**
   ```
   Launch simulation → Press 'G' → Check detected performance tier
   ```

2. **Adjust Particle Count**
   ```
   Green indicator: Can increase particle count
   Yellow indicator: Optimal balance
   Red indicator: Reduce particle count immediately
   ```

3. **Monitor CUDA Memory**
   ```
   Keep usage below 80% for stability
   If approaching limit, reduce particle count
   ```

4. **Optimize Audio Settings**
   ```
   Disable audio input if not needed
   Use "Manual Control" mode for maximum FPS
   Lower audio sensitivity reduces CPU overhead
   ```

### Resolution-Specific Recommendations

| Resolution | Min Particles | Optimal | Max Quality | Expected FPS Range |
|------------|---------------|---------|-------------|--------------------|
| 1080p      | 5,000        | 15,000  | 30,000      | 45-60+ FPS         |
| 1440p      | 8,000        | 20,000  | 40,000      | 35-55 FPS          |
| 4K         | 10,000       | 25,000  | 50,000      | 25-45 FPS          |
| 5K/8K      | 15,000       | 30,000  | 50,000      | 20-40 FPS          |

## Advanced Optimization

### GPU Driver Settings
1. **NVIDIA Control Panel**:
   - Set "Power management mode" to "Prefer maximum performance"
   - Enable "Threaded optimization"
   - Set "Texture filtering - Quality" to "High performance"

2. **Windows Settings**:
   - Set Graphics performance to "High performance"
   - Disable Windows Game Mode if experiencing stuttering
   - Close unnecessary background applications

### CUDA Optimization
- **Memory Management**: Automatic optimization included
- **Compute Capability**: Targets SM 6.1+ for best performance
- **Block Size**: Optimized for 512 threads per block
- **Memory Coalescing**: Aligned for maximum bandwidth

### OpenGL Optimization
- **Buffer Usage**: GL_DYNAMIC_DRAW for particle data
- **Vertex Array Objects**: Pre-compiled for fastest access
- **Texture Streaming**: Automatic based on GPU memory
- **Debug Output**: Can be disabled for 5-10% performance gain

## Benchmarking Your System

### Built-in Performance Monitor
```
Press 'G' → Diagnostics Panel → Monitor these metrics:
- FPS: Target >30 for smooth experience
- Frame Time: Target <33ms for good responsiveness  
- CUDA Memory: Keep <80% usage
- GPU Debug Counters: Should remain at 0
```

### Performance Testing Protocol
1. **Baseline Test**: Start with 15,000 particles
2. **Scaling Test**: Increase by 5,000 until FPS drops below 30
3. **Quality Test**: Find maximum particles with >45 FPS
4. **Stability Test**: Run for 5 minutes at chosen settings

### Common Performance Issues

#### Low FPS (<20 FPS)
- **Cause**: Too many particles for GPU capability
- **Solution**: Reduce particle count to 10,000-15,000
- **Check**: CUDA memory usage in diagnostics

#### Stuttering/Hitching
- **Cause**: CUDA memory fragmentation or Windows scheduling
- **Solution**: Restart simulation, close other GPU applications
- **Check**: Windows Task Manager for GPU usage

#### Audio Lag
- **Cause**: Audio processing overhead or device conflicts
- **Solution**: Switch to manual mode or adjust audio sensitivity
- **Check**: Try different audio input devices

## Quality vs Performance Presets

### Maximum Performance (60+ FPS)
```
Particle Count: 5,000-10,000
Quality Level: High Performance mode
Audio: Manual control recommended
Target Hardware: GTX 1060+ 
Use Case: Smooth interaction, live demonstrations
```

### Balanced Quality (30-50 FPS)
```
Particle Count: 15,000-25,000
Quality Level: Balanced mode
Audio: Live input with standard sensitivity
Target Hardware: RTX 2060+
Use Case: General scientific visualization
```

### Maximum Quality (20-35 FPS)
```
Particle Count: 35,000-50,000
Quality Level: Maximum Detail mode
Audio: High sensitivity for detailed patterns
Target Hardware: RTX 3070+
Use Case: Publication-quality visualization, research
```

### Scientific Research (Variable)
```
Particle Count: Adjusted per experiment needs
Quality Level: Based on research requirements
Audio: Calibrated for specific frequency ranges
Target Hardware: RTX 3080+ recommended
Use Case: Detailed pattern analysis, data collection
```

## Troubleshooting Performance

### Diagnostic Tools
1. **In-App Diagnostics**: Press 'G' → Diagnostics panel
2. **NVIDIA GPU-Z**: Monitor GPU utilization and memory
3. **Task Manager**: Check system resource usage
4. **Event Viewer**: Look for CUDA or OpenGL errors

### Common Solutions
- **Driver Update**: Always use latest NVIDIA drivers
- **Clean Installation**: Use DDU for complete driver reinstall
- **System Reboot**: Clear GPU memory and reset states
- **Background Apps**: Close Discord, Chrome, game launchers

### Expert Tips
- **Particle Count Sweet Spot**: Usually 1.5-2x your GPU's compute units
- **Memory Bandwidth**: Particle count limited by GPU memory bandwidth
- **Thermal Throttling**: Monitor GPU temperatures (keep <80°C)
- **Power Limits**: Ensure adequate PSU for maximum GPU performance

This guide helps you achieve optimal performance while maintaining the scientific accuracy and visual quality of the Chladni plate simulation.