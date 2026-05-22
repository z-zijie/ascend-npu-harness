#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace harness {
namespace accuracy {

struct Tolerance {
    double abs_tol;
    double rel_tol;
};

struct Mismatch {
    size_t index;
    double expected;
    double actual;
    double abs_error;
    double rel_error;
};

struct AccuracyReport {
    bool passed = false;
    size_t num_elements = 0;
    size_t mismatch_count = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    size_t max_abs_index = 0;
    size_t max_rel_index = 0;
    std::vector<Mismatch> first_mismatches;

    std::string ToString() const {
        std::ostringstream oss;
        oss << "AccuracyReport{";
        oss << "passed=" << (passed ? "true" : "false");
        oss << ", elements=" << num_elements;
        oss << ", mismatches=" << mismatch_count;
        oss << ", max_abs_err=" << max_abs_error;
        oss << " @[" << max_abs_index << "]";
        oss << ", max_rel_err=" << max_rel_error;
        oss << " @[" << max_rel_index << "]";
        if (!first_mismatches.empty()) {
            oss << ", first_mismatches=[";
            for (size_t i = 0; i < first_mismatches.size(); ++i) {
                if (i > 0) oss << ", ";
                const auto& m = first_mismatches[i];
                oss << "[" << m.index << "] exp=" << m.expected
                    << " act=" << m.actual
                    << " abs=" << m.abs_error << " rel=" << m.rel_error;
            }
            oss << "]";
        }
        oss << "}";
        return oss.str();
    }
};

template <typename T>
AccuracyReport CompareArrays(
    const T* expected,
    const T* actual,
    size_t num_elements,
    Tolerance tolerance,
    size_t max_mismatches_to_record = 16)
{
    AccuracyReport report;
    report.num_elements = num_elements;
    report.passed = true;

    for (size_t i = 0; i < num_elements; ++i) {
        double exp_val = static_cast<double>(expected[i]);
        double act_val = static_cast<double>(actual[i]);

        double abs_err = std::abs(exp_val - act_val);
        double rel_err = 0.0;
        if (std::abs(exp_val) > 1e-12) {
            rel_err = abs_err / std::abs(exp_val);
        } else if (std::abs(act_val) > 1e-12) {
            rel_err = abs_err / std::abs(act_val);
        }

        if (abs_err > tolerance.abs_tol && rel_err > tolerance.rel_tol) {
            report.mismatch_count++;
            if (abs_err > report.max_abs_error) {
                report.max_abs_error = abs_err;
                report.max_abs_index = i;
            }
            if (rel_err > report.max_rel_error) {
                report.max_rel_error = rel_err;
                report.max_rel_index = i;
            }
            if (report.first_mismatches.size() < max_mismatches_to_record) {
                report.first_mismatches.push_back(
                    Mismatch{i, exp_val, act_val, abs_err, rel_err});
            }
        }
    }

    report.passed = (report.mismatch_count == 0);
    return report;
}

// Convenience overloads for vectors
template <typename T>
AccuracyReport CompareArrays(
    const std::vector<T>& expected,
    const std::vector<T>& actual,
    Tolerance tolerance,
    size_t max_mismatches_to_record = 16)
{
    if (expected.size() != actual.size()) {
        AccuracyReport report;
        report.num_elements = 0;
        report.passed = false;
        report.mismatch_count = 1;
        report.max_abs_error = std::numeric_limits<double>::infinity();
        return report;
    }
    return CompareArrays(expected.data(), actual.data(), expected.size(),
                         tolerance, max_mismatches_to_record);
}

}  // namespace accuracy
}  // namespace harness
