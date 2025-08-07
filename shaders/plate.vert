#version 450 core

layout (location = 0) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;
uniform float frequency;
uniform float amplitude;
uniform float plate_size;

out vec3 world_pos;
out vec3 world_normal;
out vec2 tex_coord;
out float vibration_intensity;

// Chladni plate vibration function (matching the CUDA version)
float calculatePlateVibration(vec2 pos, float time, float freq, float amp) {
    float halfPlate = plate_size * 0.5;
    
    // Normalize position to [-1, 1]
    float nx = pos.x / halfPlate;
    float ny = pos.y / halfPlate;
    
    // Multiple frequency modes for richer patterns
    float baseFreq = freq * 0.008;
    
    // Primary mode
    float kx1 = baseFreq * 3.14159265;
    float ky1 = baseFreq * 3.14159265;
    
    // Secondary mode for complexity
    float kx2 = baseFreq * 3.14159265 * 1.41; // sqrt(2) ratio
    float ky2 = baseFreq * 3.14159265 * 1.73; // sqrt(3) ratio
    
    float omega = 2.0 * 3.14159265 * freq * 0.1;
    
    // Combine multiple modes
    float z1 = sin(kx1 * nx) * sin(ky1 * ny);
    float z2 = sin(kx2 * nx) * cos(ky2 * ny) * 0.5;
    
    return amp * (z1 + z2) * cos(omega * time);
}

void main() {
    // Calculate vibration displacement
    float vibration = calculatePlateVibration(position.xy, time, frequency, amplitude);
    
    // Create displaced position
    vec3 displaced_pos = position;
    displaced_pos.z += vibration;
    
    // Transform to world space
    world_pos = (model * vec4(displaced_pos, 1.0)).xyz;
    
    // Calculate normal by finite differences for proper lighting
    float epsilon = 0.01;
    float v1 = calculatePlateVibration(position.xy + vec2(epsilon, 0), time, frequency, amplitude);
    float v2 = calculatePlateVibration(position.xy + vec2(0, epsilon), time, frequency, amplitude);
    
    vec3 tangent_x = vec3(epsilon, 0, v1 - vibration);
    vec3 tangent_y = vec3(0, epsilon, v2 - vibration);
    vec3 normal = normalize(cross(tangent_x, tangent_y));
    
    world_normal = normalize((model * vec4(normal, 0.0)).xyz);
    
    // Texture coordinates
    tex_coord = (position.xy / plate_size) + 0.5; // Map to [0,1]
    
    // Store vibration intensity for fragment shader
    vibration_intensity = abs(vibration) / amplitude;
    
    gl_Position = projection * view * vec4(world_pos, 1.0);
}