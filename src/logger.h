#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <memory>

class Logger {
private:
    static std::unique_ptr<Logger> instance;
    std::ofstream logFile;
    bool logToConsole;
    bool logToFile;

    Logger() : logToConsole(true), logToFile(true) {
        if (logToFile) {
            logFile.open("chladni_debug.log", std::ios::out | std::ios::app);
            if (logFile.is_open()) {
                log("INFO", "Logger initialized - Session started");
            }
        }
    }

public:
    static Logger& getInstance() {
        if (!instance) {
            instance = std::unique_ptr<Logger>(new Logger());
        }
        return *instance;
    }

    void log(const std::string& level, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        char timestamp[64];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&time_t));
        
        std::string fullMessage = "[" + std::string(timestamp) + "." + 
                                 std::to_string(ms.count()) + "] [" + level + "] " + message;

        if (logToConsole) {
            std::cout << fullMessage << std::endl;
        }
        
        if (logToFile && logFile.is_open()) {
            logFile << fullMessage << std::endl;
            logFile.flush();
        }
    }

    void logMemory(const std::string& context, size_t bytes) {
        log("MEMORY", context + " - " + std::to_string(bytes) + " bytes");
    }

    void logCuda(const std::string& context, cudaError_t error) {
        if (error != cudaSuccess) {
            log("CUDA_ERROR", context + " - " + std::string(cudaGetErrorString(error)));
        } else {
            log("CUDA", context + " - Success");
        }
    }

    void logAudio(const std::string& context, int error = 0) {
        if (error != 0) {
            log("AUDIO_ERROR", context + " - Error code: " + std::to_string(error));
        } else {
            log("AUDIO", context + " - Success");
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            log("INFO", "Logger shutdown - Session ended");
            logFile.close();
        }
    }
};

std::unique_ptr<Logger> Logger::instance = nullptr;

// Convenience macros
#define LOG_INFO(msg) Logger::getInstance().log("INFO", msg)
#define LOG_ERROR(msg) Logger::getInstance().log("ERROR", msg)
#define LOG_WARNING(msg) Logger::getInstance().log("WARNING", msg)
#define LOG_MEMORY(context, bytes) Logger::getInstance().logMemory(context, bytes)
#define LOG_CUDA(context, error) Logger::getInstance().logCuda(context, error)
#define LOG_AUDIO(context, error) Logger::getInstance().logAudio(context, error)