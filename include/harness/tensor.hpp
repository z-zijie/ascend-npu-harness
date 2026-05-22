#pragma once
#include <vector>
#include <memory>
#include <cstring>
#include <cstdint>
#include "harness/shape.hpp"
#include "harness/dtype.hpp"

namespace harness {

template <typename T>
class Tensor {
public:
    Shape shape_;
    std::vector<T> data_;

    Tensor() = default;
    explicit Tensor(const Shape& shape) : shape_(shape), data_(shape.num_elements()) {}
    Tensor(const Shape& shape, const std::vector<T>& data) : shape_(shape), data_(data) {}

    size_t num_elements() const { return shape_.num_elements(); }
    size_t num_bytes() const { return data_.size() * sizeof(T); }
    const T* data_ptr() const { return data_.data(); }
    T* data_ptr() { return data_.data(); }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

    void copy_from(const T* src, size_t count) {
        std::memcpy(data_.data(), src, count * sizeof(T));
    }

    void copy_to(T* dst, size_t count) const {
        std::memcpy(dst, data_.data(), count * sizeof(T));
    }
};

// Aligned buffer for NPU memory
class AlignedBuffer {
public:
    void* ptr_ = nullptr;
    size_t size_ = 0;
    size_t alignment_ = 64;  // dav-2201 typically needs 64B alignment

    AlignedBuffer() = default;

    explicit AlignedBuffer(size_t size, size_t alignment = 64)
        : size_(size), alignment_(alignment)
    {
        ptr_ = std::aligned_alloc(alignment, size);
        if (!ptr_) throw std::bad_alloc();
    }

    ~AlignedBuffer() {
        if (ptr_) std::free(ptr_);
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), alignment_(other.alignment_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) std::free(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            alignment_ = other.alignment_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t size() const { return size_; }

    template <typename T>
    T* as() { return static_cast<T*>(ptr_); }

    void zero() {
        if (ptr_) std::memset(ptr_, 0, size_);
    }
};

}  // namespace harness
