#version 450 core

in vec3 world_pos;
in vec3 world_normal;
in vec2 tex_coord;
in float vibration_intensity;

out vec4 FragColor;

// Lighting uniforms
uniform vec3 light_pos;
uniform vec3 light_color;
uniform float light_intensity;
uniform vec3 ambient_color;
uniform vec3 camera_pos;

// Material properties for metal plate
uniform float roughness;
uniform float metallic;
uniform vec3 albedo;

// PBR functions (same as particle shader)
vec3 calculatePBR(vec3 albedo, float roughness, float metallic, vec3 N, vec3 V, vec3 L) {
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float HdotV = max(dot(H, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    
    // Fresnel (Schlick approximation)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);
    
    // Normal Distribution Function (GGX/Trowbridge-Reitz)
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (3.14159265 * denom * denom);
    
    // Geometry function (Smith model)
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float GL = NdotL / (NdotL * (1.0 - k) + k);
    float GV = NdotV / (NdotV * (1.0 - k) + k);
    float G = GL * GV;
    
    // Cook-Torrance BRDF
    vec3 numerator = D * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.001;
    vec3 specular = numerator / denominator;
    
    // Combine diffuse and specular
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;
    
    vec3 diffuse = kD * albedo / 3.14159265;
    
    return (diffuse + specular) * NdotL;
}

// Simple noise function for surface variation
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

void main() {
    vec3 N = normalize(world_normal);
    vec3 V = normalize(camera_pos - world_pos);
    vec3 L = normalize(light_pos - world_pos);
    
    // Base metallic material (brushed aluminum/steel)
    vec3 base_albedo = albedo;
    float base_roughness = roughness;
    float base_metallic = metallic;
    
    // Add subtle surface variation
    float surface_noise = noise(tex_coord * 200.0) * 0.05;
    base_roughness += surface_noise;
    base_roughness = clamp(base_roughness, 0.01, 0.99);
    
    // Vibration visualization - areas that vibrate more get different properties
    float vibration_effect = vibration_intensity;
    
    // Add subtle color variation based on vibration (like heat coloring on metal)
    vec3 vibration_tint = mix(vec3(1.0), vec3(1.0, 0.95, 0.9), vibration_effect * 0.3);
    base_albedo *= vibration_tint;
    
    // Calculate PBR lighting
    float light_distance = length(light_pos - world_pos);
    float attenuation = 1.0 / (1.0 + 0.05 * light_distance + 0.001 * light_distance * light_distance);
    
    vec3 lit_color = calculatePBR(base_albedo, base_roughness, base_metallic, N, V, L);
    lit_color *= light_color * light_intensity * attenuation;
    
    // Add ambient lighting
    vec3 ambient = ambient_color * base_albedo * 0.1;
    
    // Add subtle reflection for metallic surface
    vec3 reflection_dir = reflect(-V, N);
    vec3 env_color = vec3(0.3, 0.4, 0.5); // Simple sky color
    vec3 reflection = env_color * base_metallic * (1.0 - base_roughness) * 0.3;
    
    vec3 final_color = lit_color + ambient + reflection;
    
    // Add grid lines for technical appearance
    vec2 grid = abs(fract(tex_coord * 100.0) - 0.5);
    float grid_line = smoothstep(0.0, 0.02, min(grid.x, grid.y));
    final_color = mix(final_color * 0.7, final_color, grid_line);
    
    // Tone mapping and gamma correction
    final_color = final_color / (final_color + vec3(1.0));
    final_color = pow(final_color, vec3(1.0/2.2));
    
    FragColor = vec4(final_color, 1.0);
}