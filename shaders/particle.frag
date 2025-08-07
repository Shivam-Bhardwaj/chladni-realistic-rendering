#version 450 core

in vec3 world_pos;
in vec3 world_normal;
in vec2 tex_coord;
in vec3 view_dir;
in float particle_height;
in float particle_speed;

out vec4 FragColor;

// Lighting uniforms
uniform vec3 light_pos;
uniform vec3 light_color;
uniform float light_intensity;
uniform vec3 ambient_color;

// Material properties for sand/rice particles
uniform float roughness;
uniform float metallic;
uniform vec3 albedo_base;

// PBR functions
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
    float denominator = 4.0 * NdotV * NdotL + 0.001; // Add small value to prevent division by zero
    vec3 specular = numerator / denominator;
    
    // Combine diffuse and specular
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;
    
    vec3 diffuse = kD * albedo / 3.14159265;
    
    return (diffuse + specular) * NdotL;
}

void main() {
    // Create circular particle shape with soft edges
    vec2 center = vec2(0.5, 0.5);
    float dist = length(tex_coord - center);
    
    // Soft circular falloff
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    if (alpha < 0.01) discard;
    
    // Calculate surface normal for sphere-like particles
    vec3 normal;
    if (dist < 0.5) {
        // Create spherical normal
        float z = sqrt(1.0 - 4.0 * dist * dist);
        normal = normalize(vec3((tex_coord - center) * 2.0, z));
        // Transform to world space (simplified)
        normal = normalize(world_normal + normal * 0.3);
    } else {
        normal = world_normal;
    }
    
    // Dynamic particle coloring based on physics
    vec3 base_color = albedo_base;
    
    // Height-based coloring (particles higher = more energetic)
    float height_factor = clamp(particle_height * 10.0 + 0.2, 0.0, 1.0);
    vec3 height_color = mix(vec3(0.8, 0.7, 0.5), vec3(1.0, 0.9, 0.7), height_factor);
    
    // Speed-based coloring (faster particles = more red/orange)
    float speed_factor = clamp(particle_speed * 50.0, 0.0, 1.0);
    vec3 speed_color = mix(vec3(1.0), vec3(1.0, 0.6, 0.3), speed_factor);
    
    // Combine colorings
    vec3 albedo = base_color * height_color * speed_color;
    
    // Lighting calculation
    vec3 light_dir = normalize(light_pos - world_pos);
    float light_distance = length(light_pos - world_pos);
    float attenuation = 1.0 / (1.0 + 0.1 * light_distance + 0.01 * light_distance * light_distance);
    
    vec3 lit_color = calculatePBR(albedo, roughness, metallic, normal, normalize(view_dir), light_dir);
    lit_color *= light_color * light_intensity * attenuation;
    
    // Add ambient lighting
    vec3 ambient = ambient_color * albedo * 0.3;
    
    vec3 final_color = lit_color + ambient;
    
    // Add subtle subsurface scattering effect for realism
    float subsurface = pow(max(0.0, dot(-light_dir, view_dir)), 2.0) * 0.3;
    final_color += subsurface * light_color * albedo;
    
    // Tone mapping (simple Reinhard)
    final_color = final_color / (final_color + vec3(1.0));
    
    // Gamma correction
    final_color = pow(final_color, vec3(1.0/2.2));
    
    FragColor = vec4(final_color, alpha * 0.9); // Slightly transparent for realism
}