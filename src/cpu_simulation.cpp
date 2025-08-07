#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 force;
    float mass;
};

class ChladniSimulation {
private:
    GLFWwindow* window;
    std::vector<Particle> particles;
    
    // Simulation parameters
    float frequency = 440.0f;
    float amplitude = 0.005f;
    float damping = 0.1f;
    float plateSize = 2.0f;
    float gravity = 9.81f;
    float restitution = 0.5f;
    float friction = 0.3f;
    float dt = 0.001f;
    int numParticles = 5000;
    
    float simulationTime = 0.0f;
    float frequencyChangeTime = 0.0f;
    
    // Camera
    float cameraDistance = 5.0f;
    float cameraAngleX = 30.0f;
    float cameraAngleY = 45.0f;
    
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    
public:
    ChladniSimulation() : window(nullptr), rng(std::random_device{}()), dist(-1.0f, 1.0f) {}
    
    ~ChladniSimulation() {
        if (window) {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }
    
    bool init() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }
        
        window = glfwCreateWindow(1280, 720, "Chladni Plate Simulation (CPU Version)", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, this);
        
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glPointSize(3.0f);
        
        initParticles();
        
        return true;
    }
    
    void run() {
        auto lastTime = std::chrono::high_resolution_clock::now();
        
        while (!glfwWindowShouldClose(window)) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;
            
            processInput();
            updateSimulation(deltaTime);
            render();
            
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
    
private:
    void initParticles() {
        particles.resize(numParticles);
        
        for (auto& p : particles) {
            p.position = glm::vec3(
                dist(rng) * plateSize * 0.4f,
                dist(rng) * plateSize * 0.4f,
                0.1f + std::abs(dist(rng)) * 0.1f
            );
            p.velocity = glm::vec3(0.0f);
            p.force = glm::vec3(0.0f);
            p.mass = 0.001f;
        }
    }
    
    glm::vec3 calculatePlateVibration(const glm::vec3& pos, float time) {
        float kx = frequency * glm::pi<float>() / plateSize;
        float ky = frequency * glm::pi<float>() / plateSize;
        float omega = 2.0f * glm::pi<float>() * frequency;
        
        float z = amplitude * std::sin(kx * pos.x) * std::sin(ky * pos.y) * std::cos(omega * time);
        float dz_dx = amplitude * kx * std::cos(kx * pos.x) * std::sin(ky * pos.y) * std::cos(omega * time);
        float dz_dy = amplitude * ky * std::sin(kx * pos.x) * std::cos(ky * pos.y) * std::cos(omega * time);
        float dz_dt = -amplitude * omega * std::sin(kx * pos.x) * std::sin(ky * pos.y) * std::sin(omega * time);
        
        return glm::vec3(dz_dx, dz_dy, z + dz_dt);
    }
    
    void updateSimulation(float deltaTime) {
        simulationTime += deltaTime;
        frequencyChangeTime += deltaTime;
        
        // Change frequency every 5 seconds
        if (frequencyChangeTime > 5.0f) {
            frequency = 200.0f + (rand() % 800);
            frequencyChangeTime = 0.0f;
            std::cout << "Frequency changed to: " << frequency << " Hz" << std::endl;
        }
        
        // Update particles
        for (auto& p : particles) {
            // Reset force
            p.force = glm::vec3(0.0f, 0.0f, -gravity * p.mass);
            
            // Get plate vibration
            glm::vec3 plateEffect = calculatePlateVibration(p.position, simulationTime);
            
            // Check collision with plate
            if (p.position.z <= plateEffect.z + 0.001f) {
                glm::vec3 normal = glm::normalize(glm::vec3(-plateEffect.x, -plateEffect.y, 1.0f));
                float normalVel = glm::dot(p.velocity, normal);
                
                if (normalVel < 0) {
                    glm::vec3 normalForce = normal * (-normalVel * (1.0f + restitution) / dt);
                    p.force += normalForce * p.mass;
                    
                    glm::vec3 tangentVel = p.velocity - normal * normalVel;
                    float tangentSpeed = glm::length(tangentVel);
                    if (tangentSpeed > 0.0001f) {
                        glm::vec3 frictionForce = -glm::normalize(tangentVel) * friction * glm::length(normalForce);
                        p.force += frictionForce;
                    }
                }
                
                p.position.z = std::max(p.position.z, plateEffect.z + 0.001f);
            }
            
            // Add random perturbation
            p.force.x += dist(rng) * 0.01f;
            p.force.y += dist(rng) * 0.01f;
            
            // Apply damping
            p.force -= damping * p.velocity;
            
            // Update velocity and position
            glm::vec3 acceleration = p.force / p.mass;
            p.velocity += acceleration * dt;
            p.position += p.velocity * dt;
            
            // Boundary conditions
            float halfPlate = plateSize * 0.5f;
            p.position.x = glm::clamp(p.position.x, -halfPlate, halfPlate);
            p.position.y = glm::clamp(p.position.y, -halfPlate, halfPlate);
            p.position.z = std::max(0.0f, p.position.z);
        }
    }
    
    void render() {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Set up camera
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float aspect = 1280.0f / 720.0f;
        float fov = 45.0f * glm::pi<float>() / 180.0f;
        float near = 0.1f;
        float far = 100.0f;
        float top = near * std::tan(fov * 0.5f);
        float right = top * aspect;
        glFrustum(-right, right, -top, top, near, far);
        
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        glm::vec3 cameraPos = glm::vec3(
            cameraDistance * cos(glm::radians(cameraAngleY)) * cos(glm::radians(cameraAngleX)),
            cameraDistance * sin(glm::radians(cameraAngleX)),
            cameraDistance * sin(glm::radians(cameraAngleY)) * cos(glm::radians(cameraAngleX))
        );
        
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glLoadMatrixf(glm::value_ptr(view));
        
        // Draw plate grid
        glColor3f(0.3f, 0.3f, 0.3f);
        glBegin(GL_LINES);
        int gridSize = 20;
        float step = plateSize / gridSize;
        for (int i = 0; i <= gridSize; i++) {
            float pos = -plateSize/2 + i * step;
            glVertex3f(pos, -plateSize/2, 0.0f);
            glVertex3f(pos, plateSize/2, 0.0f);
            glVertex3f(-plateSize/2, pos, 0.0f);
            glVertex3f(plateSize/2, pos, 0.0f);
        }
        glEnd();
        
        // Draw particles
        glBegin(GL_POINTS);
        for (const auto& p : particles) {
            float intensity = glm::clamp(p.position.z * 10.0f + 0.5f, 0.0f, 1.0f);
            glColor3f(0.2f + 0.8f * intensity, 0.3f + 0.5f * intensity, 0.8f);
            glVertex3f(p.position.x, p.position.y, p.position.z);
        }
        glEnd();
    }
    
    void processInput() {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            cameraAngleY -= 2.0f;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            cameraAngleY += 2.0f;
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            cameraAngleX = std::min(89.0f, cameraAngleX + 2.0f);
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            cameraAngleX = std::max(-89.0f, cameraAngleX - 2.0f);
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraDistance = std::max(1.0f, cameraDistance - 0.1f);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraDistance = std::min(20.0f, cameraDistance + 0.1f);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            frequency = std::max(100.0f, frequency - 10.0f);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            frequency = std::min(2000.0f, frequency + 10.0f);
    }
};

int main() {
    ChladniSimulation sim;
    if (sim.init()) {
        std::cout << "Chladni Simulation (CPU Version)" << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  Arrow Keys: Rotate camera" << std::endl;
        std::cout << "  W/S: Zoom in/out" << std::endl;
        std::cout << "  Q/E: Decrease/Increase frequency" << std::endl;
        std::cout << "  ESC: Exit" << std::endl;
        std::cout << "\nFrequency changes automatically every 5 seconds" << std::endl;
        sim.run();
    }
    return 0;
}