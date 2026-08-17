#pragma once

#include "rfanns/types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rfanns
{
    class DenseDataset
    {
    public:
        DenseDataset() = default;
        DenseDataset(std::uint32_t count, std::uint32_t dimension, std::vector<float> values);

        static DenseDataset load(const std::string &path);

        std::uint32_t size() const noexcept { return count_; }
        std::uint32_t dimension() const noexcept { return dimension_; }
        const float *row(std::uint32_t id) const;
        const std::vector<float> &values() const noexcept { return values_; }

    private:
        std::uint32_t count_{0};
        std::uint32_t dimension_{0};
        std::vector<float> values_;
    };

    std::vector<Attribute> load_attributes(const std::string &path, std::uint32_t expected_count);
    std::vector<AttributeRange> load_ranges(const std::string &path, std::uint32_t expected_count);
    std::vector<std::vector<ExternalId>> load_groundtruth(const std::string &path,
                                                          std::uint32_t expected_queries,
                                                          std::uint32_t expected_k);

    std::vector<ExternalId> load_id_map(const std::string &path, std::uint32_t expected_count);
    void save_id_map(const std::string &path, const std::vector<ExternalId> &internal_to_external);
    void ensure_parent_directory(const std::string &path);
}

