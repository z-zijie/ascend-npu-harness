#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <numeric>

namespace harness {

struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> il) : dims(il) {}
    explicit Shape(const std::vector<int64_t>& d) : dims(d) {}

    size_t rank() const { return dims.size(); }
    int64_t dim(size_t i) const { return dims[i]; }
    const std::vector<int64_t>& data() const { return dims; }

    size_t num_elements() const {
        if (dims.empty()) return 0;
        size_t n = 1;
        for (auto d : dims) n *= static_cast<size_t>(d);
        return n;
    }

    size_t num_bytes(size_t elem_size) const {
        return num_elements() * elem_size;
    }

    bool operator==(const Shape& other) const {
        return dims == other.dims;
    }

    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << dims[i];
        }
        oss << "]";
        return oss.str();
    }

    // Index helpers
    size_t flat_index(const std::vector<int64_t>& indices) const {
        size_t idx = 0;
        size_t stride = 1;
        for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
            idx += static_cast<size_t>(indices[i]) * stride;
            stride *= static_cast<size_t>(dims[i]);
        }
        return idx;
    }

    // Validate two shapes are broadcast-compatible
    static bool broadcast_compatible(const Shape& a, const Shape& b) {
        size_t max_rank = std::max(a.rank(), b.rank());
        for (size_t i = 0; i < max_rank; ++i) {
            int64_t da = (i < a.rank()) ? a.dim(a.rank() - 1 - i) : 1;
            int64_t db = (i < b.rank()) ? b.dim(b.rank() - 1 - i) : 1;
            if (da != db && da != 1 && db != 1) return false;
        }
        return true;
    }
};

}  // namespace harness
