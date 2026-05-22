#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace harness {

enum class Dtype {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int8,
    UInt8,
};

inline const char* dtype_name(Dtype dt) {
    switch (dt) {
        case Dtype::Float32:  return "float32";
        case Dtype::Float16:  return "float16";
        case Dtype::BFloat16: return "bfloat16";
        case Dtype::Int32:    return "int32";
        case Dtype::Int8:     return "int8";
        case Dtype::UInt8:    return "uint8";
    }
    return "unknown";
}

inline size_t dtype_size(Dtype dt) {
    switch (dt) {
        case Dtype::Float32:  return 4;
        case Dtype::Float16:  return 2;
        case Dtype::BFloat16: return 2;
        case Dtype::Int32:    return 4;
        case Dtype::Int8:     return 1;
        case Dtype::UInt8:    return 1;
    }
    return 0;
}

inline Dtype dtype_from_string(const std::string& s) {
    if (s == "float32" || s == "fp32")  return Dtype::Float32;
    if (s == "float16" || s == "fp16")  return Dtype::Float16;
    if (s == "bfloat16" || s == "bf16") return Dtype::BFloat16;
    if (s == "int32")                    return Dtype::Int32;
    if (s == "int8")                     return Dtype::Int8;
    if (s == "uint8")                    return Dtype::UInt8;
    throw std::runtime_error("Unknown dtype: " + s);
}

// Default tolerances by dtype
inline double default_abs_tol(Dtype dt) {
    switch (dt) {
        case Dtype::Float32:  return 1e-5;
        case Dtype::Float16:  return 1e-2;
        case Dtype::BFloat16: return 2e-2;
        case Dtype::Int32:    return 0.0;
        case Dtype::Int8:     return 0.0;
        case Dtype::UInt8:    return 0.0;
    }
    return 1e-5;
}

inline double default_rel_tol(Dtype dt) {
    switch (dt) {
        case Dtype::Float32:  return 1e-5;
        case Dtype::Float16:  return 1e-2;
        case Dtype::BFloat16: return 2e-2;
        case Dtype::Int32:    return 0.0;
        case Dtype::Int8:     return 0.0;
        case Dtype::UInt8:    return 0.0;
    }
    return 1e-5;
}

}  // namespace harness
