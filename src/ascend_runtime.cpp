#include "harness/ascend_runtime.hpp"
#include "harness/logging.hpp"
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace harness {
namespace ascend {

// =========================================================================
// When CANN is not available, provide stub implementations
// =========================================================================

#if !defined(HARNESS_HAS_ASCEND) || HARNESS_HAS_ASCEND == 0

bool IsAvailable() { return false; }
bool RuntimeInit() { return false; }
bool RuntimeFinalize() { return false; }
int GetDeviceCount() { return 0; }
bool SetDevice(int) { return false; }
int GetCurrentDevice() { return -1; }

void StreamDeleter::operator()(void* stream) const {}
StreamHandle CreateStream() { return StreamHandle(nullptr); }
bool StreamSynchronize(void*) { return false; }

void* DeviceMalloc(size_t) { return nullptr; }
void DeviceFree(void*) {}

bool MemcpyHostToDevice(void*, const void*, size_t) { return false; }
bool MemcpyDeviceToHost(void*, const void*, size_t) { return false; }

const char* GetErrorString(int) { return "Ascend runtime not available (HARNESS_HAS_ASCEND=0)"; }
int GetLastError() { return -1; }
std::string GetDriverVersion() { return "N/A (no Ascend runtime)"; }
std::string GetSocVersion() { return "N/A"; }

void EventDeleter::operator()(void* event) const {}
EventHandle CreateEvent() { return EventHandle(nullptr); }
bool RecordEvent(void*, void*) { return false; }
bool EventElapsedTime(float*, void*, void*) { return false; }
bool EventSynchronize(void*) { return false; }

#else

// =========================================================================
// Real Ascend runtime implementation using dlopen for flexibility
// =========================================================================

static void* g_ascend_lib = nullptr;
static bool g_initialized = false;

// Function pointers
typedef int (*aclInitFunc)(const char*);
typedef int (*aclFinalizeFunc)();
typedef int (*aclrtGetDeviceCountFunc)(uint32_t*);
typedef int (*aclrtSetDeviceFunc)(int);
typedef int (*aclrtCreateStreamFunc)(void**);
typedef int (*aclrtDestroyStreamFunc)(void*);
typedef int (*aclrtSynchronizeStreamFunc)(void*);
typedef int (*aclrtMallocFunc)(void**, size_t, int);
typedef int (*aclrtFreeFunc)(void*);
typedef int (*aclrtMemcpyFunc)(void*, size_t, const void*, size_t, int);
typedef const char* (*aclGetRecentErrMsgFunc)();

static aclInitFunc p_aclInit = nullptr;
static aclFinalizeFunc p_aclFinalize = nullptr;
static aclrtGetDeviceCountFunc p_aclrtGetDeviceCount = nullptr;
static aclrtSetDeviceFunc p_aclrtSetDevice = nullptr;
static aclrtCreateStreamFunc p_aclrtCreateStream = nullptr;
static aclrtDestroyStreamFunc p_aclrtDestroyStream = nullptr;
static aclrtSynchronizeStreamFunc p_aclrtSynchronizeStream = nullptr;
static aclrtMallocFunc p_aclrtMalloc = nullptr;
static aclrtFreeFunc p_aclrtFree = nullptr;
static aclrtMemcpyFunc p_aclrtMemcpy = nullptr;
static aclGetRecentErrMsgFunc p_aclGetRecentErrMsg = nullptr;

enum { ACL_MEMCPY_HOST_TO_DEVICE = 1, ACL_MEMCPY_DEVICE_TO_HOST = 2 };

static bool load_ascend_library() {
    if (g_ascend_lib) return true;

    const char* candidates[] = {
        "libascendcl.so",
        "libacl.so",
        nullptr
    };

    for (int i = 0; candidates[i]; ++i) {
        g_ascend_lib = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);
        if (g_ascend_lib) {
            logging::info(std::string("Loaded Ascend library: ") + candidates[i]);
            break;
        }
    }

    if (!g_ascend_lib) {
        logging::warn("Cannot load Ascend runtime library; NPU features unavailable");
        return false;
    }

    #define LOAD_SYM(name) p_##name = reinterpret_cast<decltype(p_##name)>(dlsym(g_ascend_lib, #name))

    LOAD_SYM(aclInit);
    LOAD_SYM(aclFinalize);
    LOAD_SYM(aclrtGetDeviceCount);
    LOAD_SYM(aclrtSetDevice);
    LOAD_SYM(aclrtCreateStream);
    LOAD_SYM(aclrtDestroyStream);
    LOAD_SYM(aclrtSynchronizeStream);
    LOAD_SYM(aclrtMalloc);
    LOAD_SYM(aclrtFree);
    LOAD_SYM(aclrtMemcpy);
    LOAD_SYM(aclGetRecentErrMsg);

    #undef LOAD_SYM

    return true;
}

bool IsAvailable() {
    return load_ascend_library() && p_aclInit != nullptr;
}

bool RuntimeInit() {
    if (g_initialized) return true;
    if (!load_ascend_library()) return false;
    if (!p_aclInit) return false;

    int ret = p_aclInit(nullptr);
    if (ret == 0) {
        g_initialized = true;
        logging::info("Ascend runtime initialized");
        return true;
    }
    logging::error("Ascend runtime init failed: " + std::to_string(ret));
    return false;
}

bool RuntimeFinalize() {
    if (!g_initialized) return true;
    if (p_aclFinalize) p_aclFinalize();
    g_initialized = false;
    return true;
}

int GetDeviceCount() {
    if (!load_ascend_library() || !p_aclrtGetDeviceCount) {
        // Fallback: check /dev/davinci* devices
        int count = 0;
        for (int i = 0; i < 16; ++i) {
            std::string path = "/dev/davinci" + std::to_string(i);
            std::ifstream f(path);
            if (f.good()) count++;
        }
        return count;
    }
    uint32_t count = 0;
    p_aclrtGetDeviceCount(&count);
    return static_cast<int>(count);
}

bool SetDevice(int device_id) {
    if (!p_aclrtSetDevice) return false;
    return p_aclrtSetDevice(device_id) == 0;
}

int GetCurrentDevice() {
    return 0;  // Simplified; real impl would query
}

void StreamDeleter::operator()(void* stream) const {
    if (stream && p_aclrtDestroyStream) {
        p_aclrtDestroyStream(stream);
    }
}

StreamHandle CreateStream() {
    if (!p_aclrtCreateStream) return StreamHandle(nullptr);
    void* stream = nullptr;
    int ret = p_aclrtCreateStream(&stream);
    if (ret != 0) return StreamHandle(nullptr);
    return StreamHandle(stream);
}

bool StreamSynchronize(void* stream) {
    if (!p_aclrtSynchronizeStream) return false;
    return p_aclrtSynchronizeStream(stream) == 0;
}

void* DeviceMalloc(size_t size) {
    if (!p_aclrtMalloc || size == 0) return nullptr;
    void* ptr = nullptr;
    int ret = p_aclrtMalloc(&ptr, size, 0);  // ACL_MEM_MALLOC_HUGE_FIRST=0
    if (ret != 0) return nullptr;
    return ptr;
}

void DeviceFree(void* ptr) {
    if (ptr && p_aclrtFree) p_aclrtFree(ptr);
}

bool MemcpyHostToDevice(void* dst, const void* src, size_t size) {
    if (!p_aclrtMemcpy) return false;
    return p_aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE) == 0;
}

bool MemcpyDeviceToHost(void* dst, const void* src, size_t size) {
    if (!p_aclrtMemcpy) return false;
    return p_aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST) == 0;
}

const char* GetErrorString(int) {
    if (p_aclGetRecentErrMsg) return p_aclGetRecentErrMsg();
    return "Unknown error";
}

int GetLastError() {
    return 0;
}

std::string GetDriverVersion() {
    std::ifstream f("/usr/local/Ascend/driver/version.info");
    if (f.good()) {
        std::string line;
        std::getline(f, line);
        return line;
    }
    return "unknown";
}

std::string GetSocVersion() {
    return "dav-2201";
}

// Event stubs
void EventDeleter::operator()(void* event) const {}
EventHandle CreateEvent() { return EventHandle(nullptr); }
bool RecordEvent(void*, void*) { return false; }
bool EventElapsedTime(float*, void*, void*) { return false; }
bool EventSynchronize(void*) { return false; }

#endif  // HARNESS_HAS_ASCEND

}  // namespace ascend

// =========================================================================
// harness-level skip helpers
// =========================================================================
bool IsAscendRuntimeAvailable() {
    return ascend::IsAvailable();
}

bool HasUsableDevice() {
    return ascend::GetDeviceCount() > 0;
}

int GetDeviceCount() {
    return ascend::GetDeviceCount();
}

}  // namespace harness
