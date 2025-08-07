#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

// Simple logging system for the Chladni simulation

class Logger {
public:
    enum Level {
        INFO,
        WARNING,
        ERROR,
        MEMORY,
        AUDIO
    };

    static void log(Level level, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);
        
        std::ostringstream timestamp;
        timestamp << std::put_time(&tm, "%H:%M:%S");
        
        std::string level_str;
        switch (level) {
            case INFO: level_str = "INFO"; break;
            case WARNING: level_str = "WARN"; break;
            case ERROR: level_str = "ERROR"; break;
            case MEMORY: level_str = "MEM"; break;
            case AUDIO: level_str = "AUDIO"; break;
        }
        
        std::cout << "[" << timestamp.str() << "] [" << level_str << "] " << message << std::endl;
    }
};

// Convenience macros
#define LOG_INFO(msg) Logger::log(Logger::INFO, msg)
#define LOG_WARNING(msg) Logger::log(Logger::WARNING, msg)
#define LOG_ERROR(msg) Logger::log(Logger::ERROR, msg)
#define LOG_MEMORY(desc, size) Logger::log(Logger::MEMORY, std::string(desc) + " - " + std::to_string(size) + " bytes")
#define LOG_AUDIO(desc, error_code) Logger::log(Logger::AUDIO, std::string(desc) + " - code: " + std::to_string(error_code))