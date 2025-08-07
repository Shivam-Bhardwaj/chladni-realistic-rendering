#version 450 core

layout (location = 0) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 tex_coord;

void main() {
    vec4 world_pos = model * vec4(position, 1.0);
    gl_Position = projection * view * world_pos;
    
    // Map position to texture coordinates
    tex_coord = (position.xy / 15.0) + 0.5; // Assuming plate_size = 15.0
}