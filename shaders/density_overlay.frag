#version 450 core

in vec2 tex_coord;
out vec4 FragColor;

uniform sampler2D density_texture;
uniform float overlay_strength;

// Heat map color scheme
vec3 heatMapColor(float intensity) {
    intensity = clamp(intensity, 0.0, 1.0);
    
    if (intensity < 0.25) {
        // Blue to cyan
        float t = intensity / 0.25;
        return mix(vec3(0.0, 0.0, 0.5), vec3(0.0, 0.5, 1.0), t);
    } else if (intensity < 0.5) {
        // Cyan to green
        float t = (intensity - 0.25) / 0.25;
        return mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 1.0, 0.0), t);
    } else if (intensity < 0.75) {
        // Green to yellow
        float t = (intensity - 0.5) / 0.25;
        return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), t);
    } else {
        // Yellow to red
        float t = (intensity - 0.75) / 0.25;
        return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t);
    }
}

void main() {
    float density = texture(density_texture, tex_coord).r;
    
    if (density < 0.01) {
        discard; // Don't render areas with no particles
    }
    
    vec3 heat_color = heatMapColor(density);
    float alpha = density * overlay_strength;
    
    FragColor = vec4(heat_color, alpha);
}