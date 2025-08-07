#version 450 core

// Per-particle attributes
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 velocity;

// Per-instance quad vertices (for instanced rendering)
layout (location = 2) in vec3 quad_vertex;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;
uniform float particle_size;
uniform vec3 camera_pos;

out vec3 world_pos;
out vec3 world_normal;
out vec2 tex_coord;
out vec3 view_dir;
out float particle_height;
out float particle_speed;

void main() {
    // Calculate particle world position
    vec3 particle_world_pos = (model * vec4(position, 1.0)).xyz;
    
    // Create billboard facing camera
    vec3 to_camera = normalize(camera_pos - particle_world_pos);
    vec3 right = normalize(cross(vec3(0, 1, 0), to_camera));
    vec3 up = cross(to_camera, right);
    
    // Generate quad vertex position
    vec3 vertex_offset = (quad_vertex.x * right + quad_vertex.y * up) * particle_size;
    world_pos = particle_world_pos + vertex_offset;
    
    // Calculate final position
    gl_Position = projection * view * vec4(world_pos, 1.0);
    
    // Pass through data
    world_normal = to_camera; // Normal points toward camera for billboard
    tex_coord = quad_vertex.xy * 0.5 + 0.5; // Map [-1,1] to [0,1]
    view_dir = normalize(camera_pos - world_pos);
    particle_height = position.z;
    particle_speed = length(velocity);
}