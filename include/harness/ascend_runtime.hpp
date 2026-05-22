#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <memory>

namespace harness {
namespace ascend {

// =========================================================================
// RAII wrappers for Ascend runtime resources
// =========================================================================

// Initialize Ascend runtime. Call once at process start.
bool RuntimeInit();
bool RuntimeFinalize();

// Device management
int GetDeviceCount();
bool SetDevice(int device_id);
int GetCurrentDevice();

// Stream
struct StreamDeleter {
    void operator()(void* stream) const;
};
using StreamHandle = std::unique_ptr<void, StreamDeleter>;
StreamHandle CreateStream();
bool StreamSynchronize(void* stream);

// Device memory
void* DeviceMalloc(size_t size);
void DeviceFree(void* ptr);

// Memory copy
bool MemcpyHostToDevice(void* dst, const void* src, size_t size);
bool MemcpyDeviceToHost(void* dst, const void* src, size_t size);

// Error handling
const char* GetErrorString(int error_code);
int GetLastError();

// Environment check
bool IsAvailable();
std::string GetDriverVersion();
std::string GetSocVersion();

// =========================================================================
// RAII Device memory wrapper
// =========================================================================
class DeviceMemory {
public:
    DeviceMemory() = default;
    explicit DeviceMemory(size_t size) : ptr_(DeviceMalloc(size)), size_(size) {
        if (!ptr_ && size > 0) throw std::runtime_error("DeviceMalloc failed");
    }
    ~DeviceMemory() { if (ptr_) DeviceFree(ptr_); }

    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr; other.size_ = 0;
    }
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) { if (ptr_) DeviceFree(ptr_); ptr_ = other.ptr_; size_ = other.size_; other.ptr_ = nullptr; other.size_ = 0; }
        return *this;
    }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t size() const { return size_; }
    explicit operator bool() const { return ptr_ != nullptr; }

    template <typename T> T* as() { return static_cast<T*>(ptr_); }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

// =========================================================================
// Device event for timing
// =========================================================================
struct EventDeleter {
    void operator()(void* event) const;
};
using EventHandle = std::unique_ptr<void, EventDeleter>;
EventHandle CreateEvent();
bool RecordEvent(void* event, void* stream);
bool EventElapsedTime(float* ms, void* start_event, void* end_event);
bool EventSynchronize(void* event);

}  // namespace ascend
}  // namespace harness
