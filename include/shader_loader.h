#pragma once
#include <GL/glew.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class ShaderLoader {
public:
    static std::string readShaderFile(const std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open shader file: " << filePath << std::endl;
            return "";
        }
        
        std::stringstream stream;
        stream << file.rdbuf();
        file.close();
        
        return stream.str();
    }
    
    static GLuint compileShader(GLenum type, const std::string& source) {
        GLuint shader = glCreateShader(type);
        const char* src = source.c_str();
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        
        // Check compilation
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            GLchar infoLog[1024];
            glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
            std::string shaderType = (type == GL_VERTEX_SHADER) ? "VERTEX" : 
                                   (type == GL_FRAGMENT_SHADER) ? "FRAGMENT" : "COMPUTE";
            std::cerr << shaderType << " shader compilation failed:\n" << infoLog << std::endl;
            glDeleteShader(shader);
            return 0;
        }
        
        return shader;
    }

    static GLuint loadShaderProgram(const std::string& vertexPath, const std::string& fragmentPath) {
        // Read shader source code
        std::string vertexCode = readShaderFile(vertexPath);
        std::string fragmentCode = readShaderFile(fragmentPath);
        
        if (vertexCode.empty() || fragmentCode.empty()) {
            std::cerr << "Failed to read shader files" << std::endl;
            return 0;
        }
        
        // Compile shaders
        GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexCode);
        GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentCode);
        
        if (vertexShader == 0 || fragmentShader == 0) {
            std::cerr << "Failed to compile shaders" << std::endl;
            if (vertexShader != 0) glDeleteShader(vertexShader);
            if (fragmentShader != 0) glDeleteShader(fragmentShader);
            return 0;
        }
        
        // Create program and link
        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);
        
        // Check linking
        GLint success;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            GLchar infoLog[1024];
            glGetProgramInfoLog(program, 1024, nullptr, infoLog);
            std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
            glDeleteProgram(program);
            program = 0;
        }
        
        // Clean up individual shaders
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        
        std::cout << "Successfully loaded shader program: " << vertexPath << " + " << fragmentPath << std::endl;
        return program;
    }
    
};