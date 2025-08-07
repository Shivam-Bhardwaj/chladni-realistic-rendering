#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <memory>
#include "shader_loader.h"

struct RenderingConfig {
    bool enable_pbr = true;
    bool enable_shadows = true;
    bool enable_density_viz = true;
    bool enable_post_processing = true;
    
    // Material properties
    float particle_roughness = 0.8f;
    float particle_metallic = 0.1f;
    glm::vec3 particle_albedo = glm::vec3(0.9f, 0.8f, 0.6f); // Sand/rice color
    
    float plate_roughness = 0.3f;
    float plate_metallic = 0.9f;
    glm::vec3 plate_albedo = glm::vec3(0.7f, 0.7f, 0.8f); // Brushed aluminum
    
    // Lighting
    glm::vec3 light_position = glm::vec3(5.0f, 8.0f, 3.0f);
    glm::vec3 light_color = glm::vec3(1.0f, 0.95f, 0.9f); // Warm white
    float light_intensity = 3.0f;
    glm::vec3 ambient_color = glm::vec3(0.3f, 0.35f, 0.4f); // Cool ambient
    
    // Density visualization
    float density_overlay_strength = 0.4f;
    int density_texture_size = 512;
    
    // Particle rendering
    float particle_size = 0.03f; // Size of particle quads
    bool use_instanced_rendering = true;
};

class RealisticRenderer {
private:
    // Shader programs
    GLuint particle_program = 0;
    GLuint plate_program = 0;
    GLuint density_compute_program = 0;
    GLuint density_overlay_program = 0;
    
    // Particle rendering resources
    GLuint particle_VAO = 0, particle_VBO = 0, particle_instance_VBO = 0;
    GLuint quad_vertices_VBO = 0; // Quad geometry for instanced rendering
    
    // Plate rendering resources
    GLuint plate_VAO = 0, plate_VBO = 0, plate_EBO = 0;
    std::vector<float> plate_vertices;
    std::vector<unsigned int> plate_indices;
    int plate_grid_size = 200;
    
    // Density visualization resources
    GLuint density_texture = 0;
    GLuint density_overlay_VAO = 0, density_overlay_VBO = 0;
    GLuint particle_ssbo = 0; // Shader storage buffer for particles
    
    // Shadow mapping resources
    GLuint shadow_FBO = 0;
    GLuint shadow_map = 0;
    const int shadow_map_size = 2048;
    
    RenderingConfig config;
    
public:
    RealisticRenderer() = default;
    ~RealisticRenderer() { cleanup(); }
    
    bool initialize(float plate_size) {
        if (!loadShaders()) {
            std::cerr << "Failed to load shaders" << std::endl;
            return false;
        }
        
        if (!setupParticleRendering()) {
            std::cerr << "Failed to setup particle rendering" << std::endl;
            return false;
        }
        
        if (!setupPlateRendering(plate_size)) {
            std::cerr << "Failed to setup plate rendering" << std::endl;
            return false;
        }
        
        if (!setupDensityVisualization()) {
            std::cerr << "Failed to setup density visualization" << std::endl;
            return false;
        }
        
        if (!setupShadowMapping()) {
            std::cerr << "Failed to setup shadow mapping" << std::endl;
            return false;
        }
        
        return true;
    }
    
    void renderScene(const glm::mat4& view, const glm::mat4& projection, 
                    const glm::vec3& camera_pos, float time,
                    int num_particles, GLuint particle_vbo, 
                    float frequency, float amplitude, float plate_size) {
        
        // Update density visualization
        if (config.enable_density_viz) {
            updateDensityVisualization(num_particles, particle_vbo, plate_size);
        }
        
        // Render shadow map first
        if (config.enable_shadows) {
            renderShadowMap(view, projection, time, num_particles, particle_vbo, 
                           frequency, amplitude, plate_size);
        }
        
        // Main scene rendering
        glViewport(0, 0, 2560, 1440); // Reset viewport
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Render plate
        renderPlate(view, projection, camera_pos, time, frequency, amplitude, plate_size);
        
        // Render particles
        renderParticles(view, projection, camera_pos, num_particles, particle_vbo, plate_size);
        
        // Render density overlay
        if (config.enable_density_viz) {
            renderDensityOverlay(view, projection, plate_size);
        }
    }
    
    RenderingConfig& getConfig() { return config; }
    
private:
    bool loadShaders() {
        particle_program = ShaderLoader::loadShaderProgram(
            "shaders/particle.vert", "shaders/particle.frag");
        
        plate_program = ShaderLoader::loadShaderProgram(
            "shaders/plate.vert", "shaders/plate.frag");
            
        density_overlay_program = ShaderLoader::loadShaderProgram(
            "shaders/density_overlay.vert", "shaders/density_overlay.frag");
        
        // Load compute shader for density visualization
        std::string compute_source = ShaderLoader::readShaderFile("shaders/density_viz.comp");
        if (!compute_source.empty()) {
            GLuint compute_shader = ShaderLoader::compileShader(GL_COMPUTE_SHADER, compute_source);
            if (compute_shader != 0) {
                density_compute_program = glCreateProgram();
                glAttachShader(density_compute_program, compute_shader);
                glLinkProgram(density_compute_program);
                
                GLint success;
                glGetProgramiv(density_compute_program, GL_LINK_STATUS, &success);
                if (!success) {
                    glDeleteProgram(density_compute_program);
                    density_compute_program = 0;
                }
                glDeleteShader(compute_shader);
            }
        }
        
        return particle_program != 0 && plate_program != 0 && 
               density_overlay_program != 0 && density_compute_program != 0;
    }
    
    bool setupParticleRendering() {
        // Create quad vertices for instanced particle rendering
        float quad_vertices[] = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f
        };
        
        glGenVertexArrays(1, &particle_VAO);
        glGenBuffers(1, &quad_vertices_VBO);
        
        glBindVertexArray(particle_VAO);
        
        // Quad vertices (per-instance)
        glBindBuffer(GL_ARRAY_BUFFER, quad_vertices_VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(2);
        
        return true;
    }
    
    bool setupPlateRendering(float plate_size) {
        // Generate high-resolution plate mesh
        plate_vertices.clear();
        plate_indices.clear();
        
        float step = plate_size / plate_grid_size;
        
        // Generate vertices
        for (int i = 0; i <= plate_grid_size; i++) {
            for (int j = 0; j <= plate_grid_size; j++) {
                float x = -plate_size/2 + i * step;
                float y = -plate_size/2 + j * step;
                plate_vertices.push_back(x);
                plate_vertices.push_back(y);
                plate_vertices.push_back(0.0f);
            }
        }
        
        // Generate indices
        for (int i = 0; i < plate_grid_size; i++) {
            for (int j = 0; j < plate_grid_size; j++) {
                int topLeft = i * (plate_grid_size + 1) + j;
                int topRight = topLeft + 1;
                int bottomLeft = (i + 1) * (plate_grid_size + 1) + j;
                int bottomRight = bottomLeft + 1;
                
                plate_indices.push_back(topLeft);
                plate_indices.push_back(bottomLeft);
                plate_indices.push_back(topRight);
                plate_indices.push_back(topRight);
                plate_indices.push_back(bottomLeft);
                plate_indices.push_back(bottomRight);
            }
        }
        
        glGenVertexArrays(1, &plate_VAO);
        glGenBuffers(1, &plate_VBO);
        glGenBuffers(1, &plate_EBO);
        
        glBindVertexArray(plate_VAO);
        
        glBindBuffer(GL_ARRAY_BUFFER, plate_VBO);
        glBufferData(GL_ARRAY_BUFFER, plate_vertices.size() * sizeof(float), 
                     plate_vertices.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plate_EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, plate_indices.size() * sizeof(unsigned int),
                     plate_indices.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        return true;
    }
    
    bool setupDensityVisualization() {
        // Create density texture
        glGenTextures(1, &density_texture);
        glBindTexture(GL_TEXTURE_2D, density_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, config.density_texture_size, 
                     config.density_texture_size, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        // Create shader storage buffer for particles
        glGenBuffers(1, &particle_ssbo);
        
        return true;
    }
    
    bool setupShadowMapping() {
        // Create shadow map texture
        glGenTextures(1, &shadow_map);
        glBindTexture(GL_TEXTURE_2D, shadow_map);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_map_size, shadow_map_size, 
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float border_color[] = {1.0f, 1.0f, 1.0f, 1.0f};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);
        
        // Create framebuffer
        glGenFramebuffers(1, &shadow_FBO);
        glBindFramebuffer(GL_FRAMEBUFFER, shadow_FBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_map, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Shadow map framebuffer not complete" << std::endl;
            return false;
        }
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return true;
    }
    
    void renderParticles(const glm::mat4& view, const glm::mat4& projection,
                        const glm::vec3& camera_pos, int num_particles, 
                        GLuint particle_vbo, float plate_size) {
        glUseProgram(particle_program);
        
        // Set uniforms
        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(particle_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(particle_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(particle_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        
        glUniform3fv(glGetUniformLocation(particle_program, "camera_pos"), 1, glm::value_ptr(camera_pos));
        glUniform3fv(glGetUniformLocation(particle_program, "light_pos"), 1, glm::value_ptr(config.light_position));
        glUniform3fv(glGetUniformLocation(particle_program, "light_color"), 1, glm::value_ptr(config.light_color));
        glUniform1f(glGetUniformLocation(particle_program, "light_intensity"), config.light_intensity);
        glUniform3fv(glGetUniformLocation(particle_program, "ambient_color"), 1, glm::value_ptr(config.ambient_color));
        
        glUniform1f(glGetUniformLocation(particle_program, "particle_size"), config.particle_size);
        glUniform1f(glGetUniformLocation(particle_program, "roughness"), config.particle_roughness);
        glUniform1f(glGetUniformLocation(particle_program, "metallic"), config.particle_metallic);
        glUniform3fv(glGetUniformLocation(particle_program, "albedo_base"), 1, glm::value_ptr(config.particle_albedo));
        
        // Bind particle data as instance data
        glBindVertexArray(particle_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, particle_vbo);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)0); // position
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(3 * sizeof(float))); // velocity
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glVertexAttribDivisor(0, 1); // Per instance
        glVertexAttribDivisor(1, 1); // Per instance
        
        // Render instanced quads
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, num_particles);
    }
    
    void renderPlate(const glm::mat4& view, const glm::mat4& projection,
                    const glm::vec3& camera_pos, float time,
                    float frequency, float amplitude, float plate_size) {
        glUseProgram(plate_program);
        
        // Set uniforms
        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(plate_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(plate_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(plate_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        glUniform1f(glGetUniformLocation(plate_program, "time"), time);
        glUniform1f(glGetUniformLocation(plate_program, "frequency"), frequency);
        glUniform1f(glGetUniformLocation(plate_program, "amplitude"), amplitude);
        glUniform1f(glGetUniformLocation(plate_program, "plate_size"), plate_size);
        
        glUniform3fv(glGetUniformLocation(plate_program, "light_pos"), 1, glm::value_ptr(config.light_position));
        glUniform3fv(glGetUniformLocation(plate_program, "light_color"), 1, glm::value_ptr(config.light_color));
        glUniform1f(glGetUniformLocation(plate_program, "light_intensity"), config.light_intensity);
        glUniform3fv(glGetUniformLocation(plate_program, "ambient_color"), 1, glm::value_ptr(config.ambient_color));
        glUniform3fv(glGetUniformLocation(plate_program, "camera_pos"), 1, glm::value_ptr(camera_pos));
        
        glUniform1f(glGetUniformLocation(plate_program, "roughness"), config.plate_roughness);
        glUniform1f(glGetUniformLocation(plate_program, "metallic"), config.plate_metallic);
        glUniform3fv(glGetUniformLocation(plate_program, "albedo"), 1, glm::value_ptr(config.plate_albedo));
        
        // Render plate
        glBindVertexArray(plate_VAO);
        glDrawElements(GL_TRIANGLES, plate_indices.size(), GL_UNSIGNED_INT, 0);
    }
    
    void updateDensityVisualization(int num_particles, GLuint particle_vbo, float plate_size) {
        // Update particle SSBO
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, particle_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, num_particles * sizeof(float) * 8, nullptr, GL_DYNAMIC_DRAW);
        
        // Copy particle data from VBO to SSBO (simplified - in practice we'd need proper sync)
        glBindBuffer(GL_COPY_READ_BUFFER, particle_vbo);
        glBindBuffer(GL_COPY_WRITE_BUFFER, particle_ssbo);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, 
                           num_particles * sizeof(float) * 8);
        
        // Dispatch compute shader
        glUseProgram(density_compute_program);
        glBindImageTexture(0, density_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, particle_ssbo);
        
        glUniform1i(glGetUniformLocation(density_compute_program, "num_particles"), num_particles);
        glUniform1f(glGetUniformLocation(density_compute_program, "plate_size"), plate_size);
        glUniform1i(glGetUniformLocation(density_compute_program, "texture_size"), config.density_texture_size);
        
        int work_groups = (config.density_texture_size + 15) / 16;
        glDispatchCompute(work_groups, work_groups, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
    
    void renderDensityOverlay(const glm::mat4& view, const glm::mat4& projection, float plate_size) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        glUseProgram(density_overlay_program);
        
        // Use same geometry as plate but slightly elevated
        glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0.001f));
        glUniformMatrix4fv(glGetUniformLocation(density_overlay_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(density_overlay_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(density_overlay_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        glUniform1f(glGetUniformLocation(density_overlay_program, "overlay_strength"), config.density_overlay_strength);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, density_texture);
        glUniform1i(glGetUniformLocation(density_overlay_program, "density_texture"), 0);
        
        glBindVertexArray(plate_VAO);
        glDrawElements(GL_TRIANGLES, plate_indices.size(), GL_UNSIGNED_INT, 0);
        
        glDisable(GL_BLEND);
    }
    
    void renderShadowMap(const glm::mat4& view, const glm::mat4& projection, float time,
                        int num_particles, GLuint particle_vbo,
                        float frequency, float amplitude, float plate_size) {
        // TODO: Implement shadow mapping pass
        // This would render from light's perspective to create depth map
    }
    
    void cleanup() {
        if (particle_program != 0) glDeleteProgram(particle_program);
        if (plate_program != 0) glDeleteProgram(plate_program);
        if (density_compute_program != 0) glDeleteProgram(density_compute_program);
        if (density_overlay_program != 0) glDeleteProgram(density_overlay_program);
        
        if (particle_VAO != 0) glDeleteVertexArrays(1, &particle_VAO);
        if (particle_VBO != 0) glDeleteBuffers(1, &particle_VBO);
        if (quad_vertices_VBO != 0) glDeleteBuffers(1, &quad_vertices_VBO);
        
        if (plate_VAO != 0) glDeleteVertexArrays(1, &plate_VAO);
        if (plate_VBO != 0) glDeleteBuffers(1, &plate_VBO);
        if (plate_EBO != 0) glDeleteBuffers(1, &plate_EBO);
        
        if (density_texture != 0) glDeleteTextures(1, &density_texture);
        if (particle_ssbo != 0) glDeleteBuffers(1, &particle_ssbo);
        
        if (shadow_FBO != 0) glDeleteFramebuffers(1, &shadow_FBO);
        if (shadow_map != 0) glDeleteTextures(1, &shadow_map);
    }
};