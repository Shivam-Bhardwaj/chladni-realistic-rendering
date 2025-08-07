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
#include <portaudio.h>
#include <fftw3.h>
#include "logger.h"
#include "debug_info.h"
#include "cuda_memory_manager.h"

// ImGui includes
#include "imgui.h"
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

struct Particle {
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
};

struct SimulationParams {
    float frequency = 440.0f;
    float amplitude = 0.02f;  // Increased default amplitude
    float damping = 0.05f;    // Reduced damping for more movement
    float plateSize = 15.0f;  // Even larger for 5K display
    float gravity = 9.81f;
    float restitution = 0.7f; // Increased bounce
    float friction = 0.2f;    // Reduced friction
    float dt = 0.00005f;
    int numParticles = 30000;  // 3x more particles for denser, more realistic patterns
};

// External CUDA functions
extern "C" {
    void initParticles(Particle* d_particles, SimulationParams params, void* d_randStates);
    void stepSimulation(Particle* d_particles, SimulationParams params, float time, void* d_randStates);
    void resetDebugInfo();
    void fetchDebugInfo(DebugInfo* hostOut);
}

class ChladniSimulation {
public:
    // Static callback functions for GLFW
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        ChladniSimulation* sim = static_cast<ChladniSimulation*>(glfwGetWindowUserPointer(window));
        if (sim) {
            sim->handleMouseButton(button, action, mods);
        }
    }
    
    static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
        ChladniSimulation* sim = static_cast<ChladniSimulation*>(glfwGetWindowUserPointer(window));
        if (sim) {
            sim->handleMouseMove(xpos, ypos);
        }
    }
    
    static void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        ChladniSimulation* sim = static_cast<ChladniSimulation*>(glfwGetWindowUserPointer(window));
        if (sim) {
            sim->handleMouseScroll(xoffset, yoffset);
        }
    }

private:
    GLFWwindow* window;
    GLuint VAO, VBO;
    GLuint shaderProgram;
    GLuint plateVAO, plateVBO, plateEBO;
    int plateGridSize = 100;  // Store grid size for rendering
    
    CudaDevicePtr<curandState> d_randStates;
    CudaGraphicsResource cuda_vbo_resource;
    Particle* d_particles = nullptr;
    
    SimulationParams params;
    float simulationTime = 0.0f;
    
    // Audio processing
    PaStream* audioStream;
    float* audioBuffer;
    fftwf_complex* fftOutput;
    fftwf_plan fftPlan;
    static constexpr int AUDIO_BUFFER_SIZE = 1024;
    float dominantFrequency = 440.0f;
    bool audioInitialized;
    bool cudaInitialized;
    bool resourcesRegistered;
    
    // Camera
    glm::mat4 view, projection;
    float cameraDistance = 12.0f;  // Further back for larger plate
    float cameraAngleX = 30.0f;
    float cameraAngleY = 45.0f;
    
    // Mouse control
    bool firstMouse = true;
    bool mousePressed = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    float mouseSensitivity = 0.2f;
    
    // GUI control state
    bool manualMode = false;
    float manualFrequency = 440.0f;
    float manualAmplitude = 0.05f;
    bool showGUI = true;
    
public:
    ChladniSimulation() : window(nullptr), d_particles(nullptr), 
                          audioStream(nullptr), audioBuffer(nullptr), fftOutput(nullptr),
                          audioInitialized(false), cudaInitialized(false), resourcesRegistered(false) {
        LOG_INFO("ChladniSimulation constructor starting");
        
        // Initialize audio buffers with error checking
        try {
            audioBuffer = new float[AUDIO_BUFFER_SIZE]();  // Zero-initialize
            LOG_MEMORY("Audio buffer allocated", AUDIO_BUFFER_SIZE * sizeof(float));
            
            fftOutput = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (AUDIO_BUFFER_SIZE/2 + 1));
            if (!fftOutput) {
                LOG_ERROR("Failed to allocate FFTW output buffer");
                throw std::runtime_error("FFTW malloc failed");
            }
            LOG_MEMORY("FFT output buffer allocated", sizeof(fftwf_complex) * (AUDIO_BUFFER_SIZE/2 + 1));
            
            fftPlan = fftwf_plan_dft_r2c_1d(AUDIO_BUFFER_SIZE, audioBuffer, fftOutput, FFTW_ESTIMATE);
            if (!fftPlan) {
                LOG_ERROR("Failed to create FFTW plan");
                throw std::runtime_error("FFTW plan creation failed");
            }
            LOG_INFO("FFTW plan created successfully");
            
        } catch (const std::exception& e) {
            LOG_ERROR("Constructor failed: " + std::string(e.what()));
            // Cleanup partial allocations
            if (audioBuffer) { delete[] audioBuffer; audioBuffer = nullptr; }
            if (fftOutput) { fftwf_free(fftOutput); fftOutput = nullptr; }
            throw;
        }
        
        LOG_INFO("ChladniSimulation constructor completed successfully");
    }
    
    ~ChladniSimulation() {
        LOG_INFO("ChladniSimulation destructor starting");
        cleanup();
        LOG_INFO("ChladniSimulation destructor completed");
    }
    
    bool init() {
        if (!initGL()) return false;
        if (!initCUDA()) return false;
        
        // Try to initialize audio, but don't fail if it doesn't work
        if (!initAudio()) {
            LOG_WARNING("Audio initialization failed - starting in manual mode");
            std::cout << "\n⚠️  AUDIO INITIALIZATION FAILED" << std::endl;
            std::cout << "Starting simulation in manual control mode." << std::endl;
            std::cout << "Use the GUI controls (G key) to adjust frequency and amplitude." << std::endl;
            manualMode = true;  // Force manual mode
            audioInitialized = false;
        }
        
        initShaders();
        initGeometry();
        return true;
    }
    
    void run() {
        LOG_INFO("Starting simulation main loop");
        std::cout << "\n=== CHLADNI PLATE SIMULATION ===" << std::endl;
        std::cout << "🎵 Make sounds near your microphone to see particle patterns change!" << std::endl;
        std::cout << "\n🖱️ MOUSE CONTROLS:" << std::endl;
        std::cout << "  Left Click + Drag: Rotate camera around the plate" << std::endl;
        std::cout << "  Mouse Wheel: Zoom in/out" << std::endl;
        std::cout << "  Right Click: Reset camera to default position" << std::endl;
        std::cout << "  Ctrl + Mouse Wheel: Manual frequency control" << std::endl;
        std::cout << "  Shift + Mouse Wheel: Manual amplitude control" << std::endl;
        std::cout << "  ESC Key: Exit simulation" << std::endl;
        std::cout << "  G Key: Toggle GUI control panel" << std::endl;
        if (audioInitialized) {
            std::cout << "\n🔊 AUDIO:" << std::endl;
            std::cout << "  - Speak, whistle, play music, or clap near microphone" << std::endl;
            std::cout << "  - Different frequencies create different patterns" << std::endl;
            std::cout << "  - Louder sounds = more particle movement" << std::endl;
        } else {
            std::cout << "\n🎛️ MANUAL MODE (Audio unavailable):" << std::endl;
            std::cout << "  - Use GUI controls (G key) to adjust frequency and amplitude" << std::endl;
            std::cout << "  - Try different frequency presets to see patterns" << std::endl;
            std::cout << "  - Manual control gives precise parameter adjustment" << std::endl;
        }
        std::cout << "\n👀 WHAT TO LOOK FOR:" << std::endl;
        std::cout << "  - Colored particles (rice grains) on the plate" << std::endl;
        std::cout << "  - Particles move away from vibration nodes" << std::endl;
        std::cout << "  - Geometric patterns emerge based on frequency" << std::endl;
        std::cout << "\nFrequency: " << params.frequency << " Hz | Particles: " << params.numParticles << std::endl;
        std::cout << "Starting in 3 seconds..." << std::endl;
        
        auto lastTime = std::chrono::high_resolution_clock::now();
        auto startTime = lastTime;
        bool helpShown = false;
        DebugInfo dbg = {};
        
        while (!glfwWindowShouldClose(window)) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            float totalTime = std::chrono::duration<float>(currentTime - startTime).count();
            lastTime = currentTime;
            
        // Apply gentle test vibration for first few seconds
        if (totalTime < 3.0f) {
            params.amplitude = 0.01f;  // Much gentler test amplitude
            params.frequency = 440.0f;
                if (totalTime < 1.0f) {
                    std::cout << "\r🧪 Testing particle visibility with gentle vibration... " << (int)(3.0f - totalTime) << "s";
                    std::cout.flush();
                }
            } else if (totalTime >= 3.0f && totalTime < 3.5f) {
                std::cout << "\n✅ Test complete - switching to audio input mode" << std::endl;
            }
            
            // Show additional help after 8 seconds if particles might not be visible
            if (!helpShown && totalTime > 8.0f) {
                std::cout << "\n💡 TIP: If you only see a grid:" << std::endl;
                std::cout << "  - Try making LOUD sounds (whistle, speak, clap)" << std::endl;
                std::cout << "  - Use mouse wheel to zoom in/out" << std::endl;
                std::cout << "  - Left click + drag to rotate camera" << std::endl;
                std::cout << "  - Look for small colored dots moving on the plate" << std::endl;
                std::cout << "  - Try Ctrl + mouse wheel for manual high-amplitude test" << std::endl;
                helpShown = true;
            }
            
            processInput();
            updateSimulation(deltaTime);
            // Periodically fetch debug info
            static int frameCounter = 0;
            frameCounter++;
            if (frameCounter % 60 == 0) {
                fetchDebugInfo(&dbg);
                if (dbg.nan_position_count || dbg.nan_velocity_count || dbg.invalid_normal_count) {
                    LOG_WARNING(
                        std::string("GPU Debug: nan_pos=") + std::to_string(dbg.nan_position_count) +
                        ", nan_vel=" + std::to_string(dbg.nan_velocity_count) +
                        ", invalid_normal=" + std::to_string(dbg.invalid_normal_count) +
                        ", oob=" + std::to_string(dbg.oob_count) +
                        ", last_idx=" + std::to_string(dbg.last_bad_index)
                    );
                }
            }
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
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
        
        window = glfwCreateWindow(2560, 1440, "Chladni Plate Simulation - 5K Optimized", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, this);
        
        // Set mouse callbacks
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, mouseMoveCallback);
        glfwSetScrollCallback(window, mouseScrollCallback);
        
        if (glewInit() != GLEW_OK) {
            std::cerr << "Failed to initialize GLEW" << std::endl;
            return false;
        }
        
        // OpenGL debug output
        if (GLEW_KHR_debug) {
            glEnable(GL_DEBUG_OUTPUT);
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            auto glDebugCallback = [](GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
                std::string msg = std::string("GL DEBUG: ") + message;
                if (severity == GL_DEBUG_SEVERITY_HIGH) {
                    LOG_ERROR(msg);
                } else if (severity == GL_DEBUG_SEVERITY_MEDIUM) {
                    LOG_WARNING(msg);
                } else {
                    LOG_INFO(msg);
                }
            };
            glDebugMessageCallback((GLDEBUGPROC)glDebugCallback, nullptr);
            LOG_INFO("OpenGL KHR_debug enabled");
        } else {
            LOG_WARNING("OpenGL KHR_debug not available");
        }
        
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glPointSize(8.0f);  // Smaller individual points for denser particle effect
        
        // Enable vertex program point size
        glEnable(GL_PROGRAM_POINT_SIZE);
        
        // Initialize ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        
        // Setup ImGui style for 5K display
        ImGui::StyleColorsDark();
        
        // Scale UI for high-DPI displays
        ImGuiStyle& style = ImGui::GetStyle();
        style.ScaleAllSizes(2.0f);  // 2x scale for 5K
        
        // Set larger font size
        io.FontGlobalScale = 2.0f;
        
        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 450");
        
        return true;
    }
    
    bool initCUDA() {
        LOG_INFO("Initializing CUDA...");
        
        try {
            // Set CUDA device
            CUDA_CHECK(cudaSetDevice(0));
            
            // Get memory info before allocation
            auto memInfo = CudaMemoryInfo::get();
            LOG_INFO("CUDA Memory: " + std::to_string((int)memInfo.free_mb()) + "MB free / " + 
                    std::to_string((int)memInfo.total_mb()) + "MB total");
            
            // Allocate random states using RAII
            LOG_INFO("Allocating CUDA memory for " + std::to_string(params.numParticles) + " particles");
            d_randStates.allocate(params.numParticles);
            
            size_t particleSize = params.numParticles * sizeof(Particle);
            LOG_MEMORY("Particle VBO size", particleSize);
            LOG_MEMORY("Random states size", d_randStates.size());
        
            // Create VBO for particles
            LOG_INFO("Creating OpenGL buffers...");
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            
            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, particleSize, nullptr, GL_DYNAMIC_DRAW);
            
            GLenum glError = glGetError();
            if (glError != GL_NO_ERROR) {
                throw std::runtime_error("OpenGL buffer creation failed: " + std::to_string(glError));
            }
            
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
            glEnableVertexAttribArray(0);
            GLenum glAttribErr = glGetError();
            if (glAttribErr != GL_NO_ERROR) {
                throw std::runtime_error("OpenGL vertex attrib setup failed: " + std::to_string(glAttribErr));
            }
            
            // Register VBO with CUDA using RAII
            LOG_INFO("Registering VBO with CUDA...");
            cuda_vbo_resource.registerBuffer(VBO);
            
            // Map the VBO and initialize particle data directly into it
            LOG_INFO("Mapping VBO and initializing particles...");
            size_t mapped_bytes = 0;
            d_particles = (Particle*)cuda_vbo_resource.map(&mapped_bytes);
            
            LOG_INFO("Mapped VBO: " + std::to_string(mapped_bytes) + " bytes");
            
            // Initialize particles directly in VBO memory
            initParticles(d_particles, params, d_randStates.get());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Unmap after initialization
            cuda_vbo_resource.unmap();
            d_particles = nullptr; // Clear pointer since it's no longer valid
            
            resourcesRegistered = true;
            cudaInitialized = true;
            
            // Get final memory info
            auto finalMemInfo = CudaMemoryInfo::get();
            LOG_INFO("CUDA initialization complete. Memory usage: " + 
                    std::to_string((int)finalMemInfo.usage_percent()) + "% (" +
                    std::to_string((int)finalMemInfo.used_mb()) + "MB used)");
            
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR("CUDA initialization failed: " + std::string(e.what()));
            return false;
        }
    }
    
    bool initAudio() {
        LOG_INFO("Initializing PortAudio...");
        
        PaError paError = Pa_Initialize();
        LOG_AUDIO("Pa_Initialize", paError);
        if (paError != paNoError) {
            LOG_ERROR("Failed to initialize PortAudio: " + std::string(Pa_GetErrorText(paError)));
            return false;
        }
        
        // Try to find a working audio input device
        int deviceCount = Pa_GetDeviceCount();
        LOG_INFO("Found " + std::to_string(deviceCount) + " audio devices");
        
        PaDeviceIndex workingDevice = paNoDevice;
        const PaDeviceInfo* workingDeviceInfo = nullptr;
        
        // First try default input device
        PaDeviceIndex defaultInput = Pa_GetDefaultInputDevice();
        if (defaultInput != paNoDevice) {
            const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(defaultInput);
            if (deviceInfo && deviceInfo->maxInputChannels > 0) {
                LOG_INFO("Trying default input device: " + std::string(deviceInfo->name));
                if (tryAudioDevice(defaultInput, deviceInfo)) {
                    workingDevice = defaultInput;
                    workingDeviceInfo = deviceInfo;
                }
            }
        }
        
        // If default failed, try other input devices
        if (workingDevice == paNoDevice) {
            LOG_INFO("Default device failed, trying other input devices...");
            for (int i = 0; i < deviceCount; i++) {
                const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
                if (deviceInfo && deviceInfo->maxInputChannels > 0 && i != defaultInput) {
                    LOG_INFO("Trying device " + std::to_string(i) + ": " + std::string(deviceInfo->name));
                    if (tryAudioDevice(i, deviceInfo)) {
                        workingDevice = i;
                        workingDeviceInfo = deviceInfo;
                        break;
                    }
                }
            }
        }
        
        if (workingDevice == paNoDevice) {
            LOG_ERROR("No working audio input device found");
            Pa_Terminate();
            return false;
        }
        
        LOG_INFO("Successfully using audio device: " + std::string(workingDeviceInfo->name));
        audioInitialized = true;
        return true;
    }
    
private:
    bool tryAudioDevice(PaDeviceIndex deviceIndex, const PaDeviceInfo* deviceInfo) {
        PaStreamParameters inputParams;
        inputParams.device = deviceIndex;
        inputParams.channelCount = 1;
        inputParams.sampleFormat = paFloat32;
        inputParams.suggestedLatency = deviceInfo->defaultLowInputLatency;
        inputParams.hostApiSpecificStreamInfo = nullptr;
        
        PaError paError = Pa_OpenStream(&audioStream, &inputParams, nullptr, 44100, AUDIO_BUFFER_SIZE, paClipOff, 
                                       audioCallback, this);
        
        if (paError != paNoError) {
            LOG_WARNING("Failed to open stream for device: " + std::string(Pa_GetErrorText(paError)));
            return false;
        }
        
        paError = Pa_StartStream(audioStream);
        if (paError != paNoError) {
            LOG_WARNING("Failed to start stream for device: " + std::string(Pa_GetErrorText(paError)));
            Pa_CloseStream(audioStream);
            audioStream = nullptr;
            return false;
        }
        
        return true;
    }
    
public:
    
    static int audioCallback(const void* input, void* output, unsigned long frameCount,
                           const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
                           void* userData) {
        ChladniSimulation* sim = (ChladniSimulation*)userData;
        
        // Safety check
        if (!sim || !sim->audioInitialized || !sim->audioBuffer) {
            return paContinue;
        }
        
        const float* in = (const float*)input;
        if (input != nullptr) {
            unsigned long toCopy = frameCount;
            if (toCopy > (unsigned long)AUDIO_BUFFER_SIZE) {
                toCopy = AUDIO_BUFFER_SIZE;
                LOG_WARNING("Audio callback frameCount > buffer size; truncating copy");
            }
            memcpy(sim->audioBuffer, in, toCopy * sizeof(float));
            // Zero any remaining part of the buffer if fewer samples arrived
            if (toCopy < (unsigned long)AUDIO_BUFFER_SIZE) {
                memset(sim->audioBuffer + toCopy, 0, (AUDIO_BUFFER_SIZE - toCopy) * sizeof(float));
            }
            sim->processAudio();
        } else {
            // No input data provided; clear buffer to avoid processing garbage
            memset(sim->audioBuffer, 0, AUDIO_BUFFER_SIZE * sizeof(float));
        }
        
        return paContinue;
    }
    
    void processAudio() {
        // Skip audio processing if in manual mode
        if (manualMode) return;
        
        if (!fftPlan || !audioBuffer || !fftOutput) return;
        fftwf_execute(fftPlan);
        
        float maxMagnitude = 0.0f;
        int maxIndex = 0;
        
        for (int i = 1; i < AUDIO_BUFFER_SIZE/2; i++) {
            float magnitude = sqrtf(fftOutput[i][0] * fftOutput[i][0] + 
                                  fftOutput[i][1] * fftOutput[i][1]);
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
                maxIndex = i;
            }
        }
        
        dominantFrequency = (float)maxIndex * 44100.0f / AUDIO_BUFFER_SIZE;
        params.frequency = dominantFrequency;
        params.amplitude = std::min(maxMagnitude * 0.02f, 0.1f);  // Increased multiplier and max amplitude
        
        // Log audio activity every few seconds for debugging
        static auto lastAudioLog = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<float>(now - lastAudioLog).count() > 2.0f) {
            if (maxMagnitude > 0.01f) {  // Lower threshold to catch more audio
                LOG_INFO("Audio detected: Freq=" + std::to_string((int)dominantFrequency) + 
                        "Hz, Amplitude=" + std::to_string(params.amplitude) + 
                        ", RawMag=" + std::to_string(maxMagnitude));
                std::cout << "LIVE: Freq=" << (int)dominantFrequency << "Hz, Amp=" << params.amplitude << std::endl;
            }
            lastAudioLog = now;
        }
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
                gl_PointSize = 6.0;  // Smaller points for realistic dense particle effect
            }
        )";
        
        const char* fragmentShaderSource = R"(
            #version 450 core
            in float height;
            out vec4 FragColor;
            
            void main() {
                float intensity = clamp(height * 5.0 + 0.5, 0.0, 1.0);
                vec3 color = mix(vec3(0.3, 0.5, 1.0), vec3(1.0, 0.8, 0.2), intensity);
                FragColor = vec4(color, 0.9);
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
        
        plateGridSize = 200;  // Much denser grid for 5K display
        int gridSize = plateGridSize;
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
        if (!cudaInitialized || !resourcesRegistered) {
            LOG_WARNING("Simulation update called but CUDA not properly initialized");
            return;
        }
        
        simulationTime += deltaTime;
        
        try {
            // Map OpenGL buffer to CUDA using RAII
            size_t num_bytes;
            d_particles = (Particle*)cuda_vbo_resource.map(&num_bytes);
            
            // Validate buffer size
            size_t expectedBytes = params.numParticles * sizeof(Particle);
            if (num_bytes < expectedBytes) {
                LOG_WARNING("Mapped buffer size (" + std::to_string(num_bytes) + 
                           ") smaller than expected (" + std::to_string(expectedBytes) + ")");
            }
            
            // Run simulation steps - fewer substeps for performance with 30K particles
            int substeps = 2;  // Optimized for high particle count
            for (int i = 0; i < substeps; i++) {
                stepSimulation(d_particles, params, simulationTime, d_randStates.get());
                CUDA_CHECK(cudaGetLastError());
            }
            
            // Unmap buffer automatically via RAII
            cuda_vbo_resource.unmap();
            d_particles = nullptr;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Simulation update failed: " + std::string(e.what()));
            if (cuda_vbo_resource.is_mapped()) {
                cuda_vbo_resource.unmap();
                d_particles = nullptr;
            }
        }
    }
    
    void renderGUI() {
        if (!showGUI) return;
        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Control Panel Window
        ImGui::Begin("Chladni Plate Control Panel", &showGUI, ImGuiWindowFlags_AlwaysAutoResize);
        
        // Audio Mode Toggle
        ImGui::Text("Audio Control Mode:");
        int audioMode = manualMode ? 1 : 0;
        ImGui::RadioButton("Live Audio Input", &audioMode, 0); ImGui::SameLine();
        ImGui::RadioButton("Manual Control", &audioMode, 1);
        manualMode = (audioMode == 1);
        
        ImGui::Separator();
        
        if (manualMode) {
            // Manual controls
            ImGui::Text("Manual Settings:");
            ImGui::SliderFloat("Frequency (Hz)", &manualFrequency, 50.0f, 2000.0f, "%.0f Hz");
            ImGui::SliderFloat("Amplitude", &manualAmplitude, 0.0f, 0.15f, "%.3f");
            
            ImGui::Separator();
            
            // Preset frequency buttons
            ImGui::Text("Quick Frequency Presets:");
            if (ImGui::Button("100 Hz")) manualFrequency = 100.0f;
            ImGui::SameLine();
            if (ImGui::Button("200 Hz")) manualFrequency = 200.0f;
            ImGui::SameLine();
            if (ImGui::Button("440 Hz")) manualFrequency = 440.0f;
            ImGui::SameLine();
            if (ImGui::Button("800 Hz")) manualFrequency = 800.0f;
            
            if (ImGui::Button("1200 Hz")) manualFrequency = 1200.0f;
            ImGui::SameLine();
            if (ImGui::Button("1600 Hz")) manualFrequency = 1600.0f;
            ImGui::SameLine();
            if (ImGui::Button("2000 Hz")) manualFrequency = 2000.0f;
            
            ImGui::Separator();
            
            // Quick amplitude buttons
            ImGui::Text("Quick Amplitude Presets:");
            if (ImGui::Button("Low (0.02)")) manualAmplitude = 0.02f;
            ImGui::SameLine();
            if (ImGui::Button("Medium (0.05)")) manualAmplitude = 0.05f;
            ImGui::SameLine();
            if (ImGui::Button("High (0.1)")) manualAmplitude = 0.1f;
            
            // Apply manual settings to simulation
            params.frequency = manualFrequency;
            params.amplitude = manualAmplitude;
        } else {
            // Show current audio input
            ImGui::Text("Live Audio Status:");
            ImGui::Text("Frequency: %.0f Hz", dominantFrequency);
            ImGui::Text("Amplitude: %.3f", params.amplitude);
            
            // Audio sensitivity adjustment
            ImGui::Separator();
            static float audioSensitivity = 0.02f;
            ImGui::SliderFloat("Audio Sensitivity", &audioSensitivity, 0.001f, 0.1f, "%.3f");
            ImGui::Text("Adjust if particles don't respond to sound");
        }
        
        ImGui::Separator();
        
        // Comprehensive diagnostics
        ImGui::Separator();
        ImGui::Text("🔧 DIAGNOSTICS");
        
        // Performance metrics
        ImGui::Text("⚡ Performance:");
        ImGui::Text("  FPS: %.1f (%.2fms frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::Text("  Simulation Time: %.2fs", simulationTime);
        
        // Simulation parameters
        ImGui::Text("🎛️ Simulation:");
        ImGui::Text("  Particles: %d", params.numParticles);
        ImGui::Text("  Plate Size: %.1f units", params.plateSize);
        ImGui::Text("  Time Step: %.6f", params.dt);
        ImGui::Text("  Damping: %.3f", params.damping);
        ImGui::Text("  Gravity: %.1f", params.gravity);
        
        // Current physics state
        ImGui::Text("📊 Current State:");
        ImGui::Text("  Frequency: %.0f Hz", params.frequency);
        ImGui::Text("  Amplitude: %.4f", params.amplitude);
        if (audioInitialized && !manualMode) {
            ImGui::Text("  Dominant Audio Freq: %.0f Hz", dominantFrequency);
        }
        
        // Memory diagnostics
        if (cudaInitialized) {
            try {
                auto memInfo = CudaMemoryInfo::get();
                ImGui::Text("💾 CUDA Memory:");
                ImGui::Text("  Used: %.0f MB (%.1f%%)", memInfo.used_mb(), memInfo.usage_percent());
                ImGui::Text("  Free: %.0f MB", memInfo.free_mb());
                ImGui::Text("  Total: %.0f MB", memInfo.total_mb());
            } catch (...) {
                ImGui::Text("💾 CUDA Memory: Query failed");
            }
        }
        
        // System status
        ImGui::Text("🖥️ System Status:");
        ImGui::Text("  CUDA: %s", cudaInitialized ? "✅ OK" : "❌ FAIL");
        ImGui::Text("  Audio: %s", audioInitialized ? "✅ OK" : "❌ FAIL");
        ImGui::Text("  Graphics: %s", resourcesRegistered ? "✅ OK" : "❌ FAIL");
        
        // Debug info from GPU
        static DebugInfo dbg = {};
        static int debugUpdateCounter = 0;
        debugUpdateCounter++;
        if (debugUpdateCounter % 30 == 0) { // Update every 30 frames (~0.5s at 60fps)
            fetchDebugInfo(&dbg);
        }
        
        ImGui::Text("🐛 GPU Debug:");
        ImGui::Text("  NaN Positions: %u", dbg.nan_position_count);
        ImGui::Text("  NaN Velocities: %u", dbg.nan_velocity_count);
        ImGui::Text("  Invalid Normals: %u", dbg.invalid_normal_count);
        if (dbg.oob_count > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "  ⚠️ Out of Bounds: %u (auto-reset)", dbg.oob_count);
        } else {
            ImGui::Text("  ✅ Out of Bounds: 0");
        }
        if (dbg.last_bad_index > 0) {
            ImGui::Text("  Last Bad Index: %u", dbg.last_bad_index);
        }
        
        // Reset debug counters button
        if (ImGui::Button("🔄 Reset Debug Counters")) {
            resetDebugInfo();
        }
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "(Resets GPU error counts)");
        
        // Camera controls info
        ImGui::Separator();
        ImGui::Text("Camera Controls:");
        ImGui::BulletText("Left Click + Drag: Rotate");
        ImGui::BulletText("Mouse Wheel: Zoom");
        ImGui::BulletText("Right Click: Reset Camera");
        
        // Toggle GUI visibility
        ImGui::Separator();
        if (ImGui::Button("Hide Controls (Press G to show)")) {
            showGUI = false;
        }
        
        ImGui::End();
        
        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
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
        projection = glm::perspective(glm::radians(45.0f), 2560.0f/1440.0f, 0.1f, 100.0f);
        
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        // Draw plate
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glBindVertexArray(plateVAO);
        glDrawElements(GL_TRIANGLES, 6 * plateGridSize * plateGridSize, GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
        // Draw particles
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, params.numParticles);
        
        // Render GUI
        renderGUI();
        
        // Debug: Log render call occasionally
        static int renderCount = 0;
        renderCount++;
        if (renderCount % 120 == 0) {  // Every 2 seconds to reduce spam with more particles
            std::cout << "Rendered " << params.numParticles << " particles on " << plateGridSize << "x" << plateGridSize << " grid" << std::endl;
        }
    }
    
    void processInput() {
        // ESC to exit
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
            
        // G to toggle GUI
        static bool gKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
            if (!gKeyPressed) {
                showGUI = !showGUI;
                gKeyPressed = true;
            }
        } else {
            gKeyPressed = false;
        }
    }
    
    // Mouse handler methods
    void handleMouseButton(int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                mousePressed = true;
                glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
                firstMouse = true;  // Reset first mouse to avoid jumps
            } else if (action == GLFW_RELEASE) {
                mousePressed = false;
            }
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
            // Right click - reset camera to default position
            cameraDistance = 12.0f;  // Reset to new default
            cameraAngleX = 30.0f;
            cameraAngleY = 45.0f;
            std::cout << "Camera reset to default position" << std::endl;
        }
    }
    
    void handleMouseMove(double xpos, double ypos) {
        if (!mousePressed) return;
        
        if (firstMouse) {
            lastMouseX = xpos;
            lastMouseY = ypos;
            firstMouse = false;
            return;
        }
        
        double xoffset = xpos - lastMouseX;
        double yoffset = lastMouseY - ypos; // Reversed: y-coordinates go from bottom to top
        
        lastMouseX = xpos;
        lastMouseY = ypos;
        
        // Apply mouse sensitivity
        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;
        
        // Update camera angles
        cameraAngleY += (float)xoffset;
        cameraAngleX += (float)yoffset;
        
        // Constrain the camera angles
        if (cameraAngleX > 89.0f) cameraAngleX = 89.0f;
        if (cameraAngleX < -89.0f) cameraAngleX = -89.0f;
        
        // Allow full 360 degree rotation horizontally
        if (cameraAngleY > 360.0f) cameraAngleY -= 360.0f;
        if (cameraAngleY < 0.0f) cameraAngleY += 360.0f;
    }
    
    void handleMouseScroll(double xoffset, double yoffset) {
        // Check if Ctrl is held for frequency control, Shift for amplitude control
        bool ctrlPressed = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || 
                          glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
        bool shiftPressed = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || 
                           glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        
        if (ctrlPressed) {
            // Control frequency with Ctrl + scroll
            float freqChange = (float)yoffset * 50.0f;
            params.frequency = std::max(100.0f, std::min(2000.0f, params.frequency + freqChange));
            std::cout << "Frequency: " << (int)params.frequency << "Hz" << std::endl;
        } else if (shiftPressed) {
            // Control amplitude with Shift + scroll
            float ampChange = (float)yoffset * 0.01f;  // Increased sensitivity
            params.amplitude = std::max(0.0f, std::min(0.2f, params.amplitude + ampChange));  // Higher max
            std::cout << "Manual Amplitude: " << params.amplitude << " (try values around 0.05-0.1)" << std::endl;
        } else {
            // Normal zoom control
            float zoomFactor = 1.0f - (float)yoffset * 0.1f;
            cameraDistance *= zoomFactor;
            cameraDistance = std::max(1.0f, std::min(20.0f, cameraDistance));
        }
    }
    
    void cleanup() {
        LOG_INFO("Starting cleanup...");
        
        // Audio cleanup - stop stream first, then close
        if (audioInitialized) {
            LOG_INFO("Cleaning up audio...");
            
            if (audioStream) {
                PaError paError = Pa_StopStream(audioStream);
                LOG_AUDIO("Pa_StopStream", paError);
                
                paError = Pa_CloseStream(audioStream);
                LOG_AUDIO("Pa_CloseStream", paError);
                audioStream = nullptr;
            }
            
            Pa_Terminate();
            LOG_AUDIO("Pa_Terminate", 0);
            audioInitialized = false;
        }
        
        // CUDA cleanup - RAII handles resource cleanup automatically
        if (cudaInitialized) {
            LOG_INFO("Cleaning up CUDA...");
            
            // Unmap if still mapped
            if (cuda_vbo_resource.is_mapped()) {
                cuda_vbo_resource.unmap();
                d_particles = nullptr;
            }
            
            // RAII destructors will handle cleanup automatically
            cuda_vbo_resource = CudaGraphicsResource{};
            d_randStates = CudaDevicePtr<curandState>{};
            d_particles = nullptr;
            
            resourcesRegistered = false;
            cudaInitialized = false;
            LOG_INFO("CUDA cleanup completed via RAII");
        }
        
        // Audio buffer cleanup
        if (audioBuffer) {
            LOG_INFO("Cleaning up audio buffer...");
            delete[] audioBuffer;
            audioBuffer = nullptr;
        }
        
        if (fftPlan) {
            LOG_INFO("Destroying FFTW plan...");
            fftwf_destroy_plan(fftPlan);
            fftPlan = nullptr;
        }
        
        if (fftOutput) {
            LOG_INFO("Freeing FFTW output buffer...");
            fftwf_free(fftOutput);
            fftOutput = nullptr;
        }
        
        // OpenGL cleanup
        LOG_INFO("Cleaning up OpenGL resources...");
        if (VAO != 0) { glDeleteVertexArrays(1, &VAO); VAO = 0; }
        if (VBO != 0) { glDeleteBuffers(1, &VBO); VBO = 0; }
        if (plateVAO != 0) { glDeleteVertexArrays(1, &plateVAO); plateVAO = 0; }
        if (plateVBO != 0) { glDeleteBuffers(1, &plateVBO); plateVBO = 0; }
        if (plateEBO != 0) { glDeleteBuffers(1, &plateEBO); plateEBO = 0; }
        if (shaderProgram != 0) { glDeleteProgram(shaderProgram); shaderProgram = 0; }
        
        // ImGui cleanup
        LOG_INFO("Cleaning up ImGui...");
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        // Window cleanup
        if (window) {
            LOG_INFO("Cleaning up GLFW...");
            glfwDestroyWindow(window);
            glfwTerminate();
            window = nullptr;
        }
        
        LOG_INFO("Cleanup completed successfully");
    }
};

int main() {
    ChladniSimulation sim;
    if (sim.init()) {
        sim.run();
    }
    return 0;
}