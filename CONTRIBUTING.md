# Contributing to Chladni Realistic Rendering

Thank you for your interest in contributing to the Chladni Realistic Rendering project! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** following the README
4. **Create a branch** for your feature or bugfix
5. **Make your changes** following our coding standards
6. **Test thoroughly** before submitting
7. **Submit a pull request**

## Areas for Contribution

### High Priority
- **Shadow Mapping**: Complete the shadow mapping system in the realistic renderer
- **Post-Processing Effects**: Bloom, anti-aliasing, tone mapping
- **Performance Optimization**: GPU profiling and optimization
- **Platform Support**: macOS and Linux compatibility improvements

### Medium Priority
- **Additional Visualization Modes**: New ways to visualize particle data
- **VR/AR Support**: OpenXR integration for immersive experiences
- **Advanced Materials**: More realistic surface properties
- **Animation Export**: Video recording capabilities

### Documentation & Examples
- **Tutorials**: Step-by-step guides for specific features
- **API Documentation**: Comprehensive code documentation
- **Example Projects**: Additional demo applications
- **Performance Guides**: Optimization best practices

## Development Guidelines

### Code Style
- **C++17** standard with modern practices
- **RAII** for all resource management
- **Consistent naming**: camelCase for variables, PascalCase for classes
- **Documentation**: Document public APIs and complex algorithms

### CUDA Guidelines
- **Memory Management**: Always use RAII wrappers
- **Error Checking**: Use CUDA_CHECK macro for all CUDA calls
- **Performance**: Profile with Nsight for optimization
- **Compute Capability**: Support SM 6.0+ (Pascal architecture and newer)

### Shader Guidelines
- **Version**: Use GLSL 4.5+ for compute shader compatibility
- **Performance**: Optimize for modern GPUs (avoid excessive branching)
- **Readability**: Comment complex mathematical operations
- **Testing**: Test on multiple GPU vendors when possible

### Commit Guidelines
- **Descriptive messages**: Clear, concise commit messages
- **Atomic commits**: One logical change per commit
- **Testing**: Ensure all tests pass before committing
- **Dependencies**: Update documentation if dependencies change

## Testing

### Before Submitting
1. **Build tests**: Ensure the project builds on Windows
2. **Runtime tests**: Test with different GPU configurations
3. **Performance tests**: Verify no significant performance regression
4. **Memory tests**: Check for leaks using debugging tools

### Test Environment
- **Primary**: Windows 10/11 with NVIDIA GPU
- **Secondary**: Test on different CUDA compute capabilities if possible
- **Audio**: Test both with and without audio input devices

## Pull Request Process

1. **Create descriptive title** and detailed description
2. **Reference issues** that your PR addresses
3. **Include screenshots/videos** for visual changes
4. **Update documentation** if adding new features
5. **Ensure CI passes** (when available)
6. **Respond to review feedback** promptly

### PR Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Local testing completed
- [ ] Performance benchmarked
- [ ] Multiple GPU configurations tested

## Screenshots/Videos
(Include for visual changes)
```

## Bug Reports

### Information to Include
- **System specifications**: GPU, OS, driver versions
- **Build configuration**: Debug/Release, CMake version
- **Reproduction steps**: Clear steps to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs/Output**: Relevant error messages or logs

### Bug Report Template
```markdown
**System Information:**
- GPU: [e.g., RTX 3070]
- OS: [e.g., Windows 11]
- CUDA Version: [e.g., 12.0]
- Driver Version: [e.g., 526.98]

**Describe the Bug:**
A clear description of what the bug is.

**Reproduction Steps:**
1. Step one
2. Step two
3. See error

**Expected Behavior:**
What you expected to happen.

**Screenshots/Logs:**
If applicable, add screenshots or log output.
```

## Feature Requests

### Information to Include
- **Use case**: Why is this feature needed?
- **Description**: Detailed description of the proposed feature
- **Implementation ideas**: Any thoughts on how it could be implemented
- **Alternatives**: Alternative solutions you've considered

## Architecture Guidelines

### Adding New Renderers
1. **Inherit from base interfaces** when possible
2. **Follow RAII patterns** for resource management
3. **Add configuration options** to RenderingConfig
4. **Include performance metrics** for comparison

### Shader Development
1. **Separate files**: Keep vertex, fragment, compute shaders in separate files
2. **Uniform blocks**: Group related uniforms for better performance
3. **Error handling**: Provide clear error messages for compilation failures
4. **Documentation**: Comment complex shader algorithms

### CUDA Kernel Development
1. **Block size optimization**: Profile different block sizes
2. **Memory coalescing**: Ensure optimal memory access patterns
3. **Occupancy**: Use CUDA profiler to optimize occupancy
4. **Error checking**: Always check kernel launch errors

## Performance Considerations

### Benchmarking
- **Consistent environment**: Use same hardware/settings for comparisons
- **Multiple runs**: Average results over multiple runs
- **Different scenarios**: Test with various particle counts and frequencies
- **Memory usage**: Monitor GPU memory consumption

### Optimization Targets
- **60 FPS minimum** at 1920x1080 with 30K particles (RTX 3070)
- **Memory efficiency**: < 200MB GPU memory usage
- **Startup time**: < 3 seconds initialization time
- **CPU usage**: < 20% on modern 8-core systems

## Code Review Process

### For Contributors
- **Self-review**: Review your own code before submitting
- **Clean history**: Squash commits if necessary for cleaner history
- **Documentation**: Update relevant documentation
- **Testing**: Include test results in PR description

### For Reviewers
- **Constructive feedback**: Provide specific, actionable feedback
- **Performance focus**: Consider performance implications
- **Security**: Review for potential security issues
- **Compatibility**: Consider impact on different hardware/OS

## Resources

### Learning Resources
- **CUDA Programming**: [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- **OpenGL**: [LearnOpenGL.com](https://learnopengl.com/)
- **PBR Theory**: [Real Shading in Unreal Engine 4](https://blog.selfshadow.com/publications/s2013-shading-course/)
- **CMake**: [Modern CMake](https://cliutils.gitlab.io/modern-cmake/)

### Tools
- **NVIDIA Nsight Graphics**: GPU debugging and profiling
- **RenderDoc**: Graphics debugging
- **Visual Studio**: Primary IDE for Windows development
- **Git**: Version control best practices

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion  
- **Code Review**: For implementation questions during PR process

Thank you for contributing to making scientific visualization more accessible and beautiful!