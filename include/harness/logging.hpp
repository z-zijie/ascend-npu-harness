#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace harness {
namespace logging {

enum class Level { Debug = 0, Info = 1, Warn = 2, Error = 3 };

inline const char* level_str(Level lv) {
    switch (lv) {
        case Level::Debug: return "DEBUG";
        case Level::Info:  return "INFO";
        case Level::Warn:  return "WARN";
        case Level::Error: return "ERROR";
    }
    return "?";
}

inline std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

inline void log(Level lv, const std::string& msg) {
    std::cerr << "[" << timestamp() << "][" << level_str(lv) << "] " << msg << std::endl;
}

inline void debug(const std::string& msg) { log(Level::Debug, msg); }
inline void info(const std::string& msg)  { log(Level::Info, msg); }
inline void warn(const std::string& msg)  { log(Level::Warn, msg); }
inline void error(const std::string& msg) { log(Level::Error, msg); }

}  // namespace logging
}  // namespace harness
