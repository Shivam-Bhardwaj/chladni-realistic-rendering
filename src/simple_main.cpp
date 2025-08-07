#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

struct Particle {
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
};

struct SimulationParams {
    float frequency = 440.0f;
    float amplitude = 0.005f;
    float damping = 0.1f;
    float plateSize = 2.0f;
    float gravity = 9.81f;
    float restitution = 0.5f;
    float friction = 0.3f;
    float dt = 0.0001f;
    int numParticles = 50000;
};

// External CUDA functions
extern "C" {
    void initParticles(Particle* d_particles, SimulationParams params, void* d_randStates);
    void stepSimulation(Particle* d_particles, SimulationParams params, float time, void* d_randStates);
}

class ChladniSimulation {
private:
    GLFWwindow* window;
    GLuint VAO, VBO;
    GLuint shaderProgram;
    GLuint plateVAO, plateVBO, plateEBO;
    
    Particle* d_particles;
    void* d_randStates;
    cudaGraphicsResource* cuda_vbo_resource;
    
    SimulationParams params;
    float simulationTime = 0.0f;
    float frequencyChangeTime = 0.0f;
    
    // Camera
    glm::mat4 view, projection;
    float cameraDistance = 5.0f;
    float cameraAngleX = 30.0f;
    float cameraAngleY = 45.0f;
    
public:
    ChladniSimulation() : window(nullptr), d_particles(nullptr), d_randStates(nullptr) {}
    
    ~ChladniSimulation() {
        cleanup();
    }
    
    bool init() {
        if (!initGL()) return false;
        if (!initCUDA()) return false;
        initShaders();
        initGeometry();
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
    bool initGL() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        window = glfwCreateWindow(1920, 1080, "Chladni Plate Simulation (No Audio)", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, this);
        
        if (glewInit() != GLEW_OK) {
            std::cerr << "Failed to initialize GLEW" << std::endl;
            return false;
        }
        
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glPointSize(2.0f);
        
        return true;
    }
    
    bool initCUDA() {
        cudaSetDevice(0);
        
        // Allocate particle memory
        size_t particleSize = params.numParticles * sizeof(Particle);
        cudaMalloc(&d_particles, particleSize);
        
        // Allocate random states
        size_t randStateSize = params.numParticles * sizeof(curandState);
        cudaMalloc(&d_randStates, randStateSize);
        
        // Initialize particles
        initParticles(d_particles, params, d_randStates);
        
        // Create VBO for particles
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, particleSize, nullptr, GL_DYNAMIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Register VBO with CUDA
        cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsRegisterFlagsNone);
        
        return true;
    }
    
    void initShaders() {
        const char* vertexShaderSource = R"(
            #version 450 core
            layout (location = 0) in vec3 aPos;
            
            uniform mat4 view;
            uniform mat4 projection;
            
            out float height;
            
            void main() {
                gl_Position = projection * view * vec4(aPos, 1.0);
                height = aPos.z;
            }
        )";
        
        const char* fragmentShaderSource = R"(
            #version 450 core
            in float height;
            out vec4 FragColor;
            
            void main() {
                float intensity = clamp(height * 10.0 + 0.5, 0.0, 1.0);
                vec3 color = mix(vec3(0.2, 0.3, 0.8), vec3(1.0, 0.8, 0.2), intensity);
                FragColor = vec4(color, 0.8);
            }
        )";
        
        GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
        GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
        
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
    
    GLuint compileShader(GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        return shader;
    }
    
    void initGeometry() {
        // Create plate mesh
        std::vector<float> plateVertices;
        std::vector<unsigned int> plateIndices;
        
        int gridSize = 50;
        float step = params.plateSize / gridSize;
        
        for (int i = 0; i <= gridSize; i++) {
            for (int j = 0; j <= gridSize; j++) {
                float x = -params.plateSize/2 + i * step;
                float y = -params.plateSize/2 + j * step;
                plateVertices.push_back(x);
                plateVertices.push_back(y);
                plateVertices.push_back(0.0f);
            }
        }
        
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                int topLeft = i * (gridSize + 1) + j;
                int topRight = topLeft + 1;
                int bottomLeft = (i + 1) * (gridSize + 1) + j;
                int bottomRight = bottomLeft + 1;
                
                plateIndices.push_back(topLeft);
                plateIndices.push_back(bottomLeft);
                plateIndices.push_back(topRight);
                plateIndices.push_back(topRight);
                plateIndices.push_back(bottomLeft);
                plateIndices.push_back(bottomRight);
            }
        }
        
        glGenVertexArrays(1, &plateVAO);
        glGenBuffers(1, &plateVBO);
        glGenBuffers(1, &plateEBO);
        
        glBindVertexArray(plateVAO);
        glBindBuffer(GL_ARRAY_BUFFER, plateVBO);
        glBufferData(GL_ARRAY_BUFFER, plateVertices.size() * sizeof(float), 
                     plateVertices.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plateEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, plateIndices.size() * sizeof(unsigned int), 
                     plateIndices.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }
    
    void updateSimulation(float deltaTime) {
        simulationTime += deltaTime;
        frequencyChangeTime += deltaTime;
        
        // Automatically change frequency every 5 seconds
        if (frequencyChangeTime > 5.0f) {
            params.frequency = 200.0f + (rand() % 800);  // Random frequency between 200-1000 Hz
            frequencyChangeTime = 0.0f;
            std::cout << "Frequency changed to: " << params.frequency << " Hz" << std::endl;
        }
        
        // Map OpenGL buffer to CUDA
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &num_bytes, cuda_vbo_resource);
        
        // Run simulation steps
        int substeps = 10;
        for (int i = 0; i < substeps; i++) {
            stepSimulation(d_particles, params, simulationTime, d_randStates);
        }
        
        // Unmap buffer
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    }
    
    void render() {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glUseProgram(shaderProgram);
        
        // Set up camera
        glm::vec3 cameraPos = glm::vec3(
            cameraDistance * cos(glm::radians(cameraAngleY)) * cos(glm::radians(cameraAngleX)),
            cameraDistance * sin(glm::radians(cameraAngleX)),
            cameraDistance * sin(glm::radians(cameraAngleY)) * cos(glm::radians(cameraAngleX))
        );
        
        view = glm::lookAt(cameraPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        projection = glm::perspective(glm::radians(45.0f), 1920.0f/1080.0f, 0.1f, 100.0f);
        
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        // Draw plate
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glBindVertexArray(plateVAO);
        glDrawElements(GL_TRIANGLES, 6 * 50 * 50, GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
        // Draw particles
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, params.numParticles);
    }
    
    void processInput() {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            cameraAngleY -= 1.0f;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            cameraAngleY += 1.0f;
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            cameraAngleX = std::min(89.0f, cameraAngleX + 1.0f);
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            cameraAngleX = std::max(-89.0f, cameraAngleX - 1.0f);
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraDistance = std::max(1.0f, cameraDistance - 0.1f);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraDistance = std::min(20.0f, cameraDistance + 0.1f);
        
        // Manual frequency control
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            params.frequency = std::max(100.0f, params.frequency - 10.0f);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            params.frequency = std::min(2000.0f, params.frequency + 10.0f);
    }
    
    void cleanup() {
        if (cuda_vbo_resource) {
            cudaGraphicsUnregisterResource(cuda_vbo_resource);
        }
        
        if (d_particles) cudaFree(d_particles);
        if (d_randStates) cudaFree(d_randStates);
        
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteVertexArrays(1, &plateVAO);
        glDeleteBuffers(1, &plateVBO);
        glDeleteBuffers(1, &plateEBO);
        glDeleteProgram(shaderProgram);
        
        if (window) {
            glfwDestroyWindow(window);
            glfwTerminate();
        }
    }
};

int main() {
    ChladniSimulation sim;
    if (sim.init()) {
        std::cout << "Chladni Simulation Started (No Audio Version)" << std::endl;
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