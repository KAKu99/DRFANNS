#include "rfanns/dataset.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace rfanns
{
    namespace
    {
        template <typename T>
        void read_exact(std::ifstream &input, T *destination, std::size_t count, const std::string &what)
        {
            input.read(reinterpret_cast<char *>(destination),
                       static_cast<std::streamsize>(count * sizeof(T)));
            if (!input)
                throw Error("file ended while reading " + what);
        }
    }

    DenseDataset::DenseDataset(std::uint32_t count,
                               std::uint32_t dimension,
                               std::vector<float> values)
        : count_(count), dimension_(dimension), values_(std::move(values))
    {
        if (count_ == 0 || dimension_ == 0)
            throw Error("DenseDataset count and dimension must be positive");
        if (values_.size() != static_cast<std::size_t>(count_) * dimension_)
            throw Error("DenseDataset value count does not match count * dimension");
    }

    DenseDataset DenseDataset::load(const std::string &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw Error("cannot open vector file " + path);

        std::int32_t count = 0;
        std::int32_t dimension = 0;
        read_exact(input, &count, 1, "vector count");
        read_exact(input, &dimension, 1, "vector dimension");
        if (count <= 0 || dimension <= 0)
            throw Error("vector file has a non-positive count or dimension: " + path);

        const std::size_t total = static_cast<std::size_t>(count) * static_cast<std::size_t>(dimension);
        if (total > std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw Error("vector file size overflows addressable memory: " + path);
        std::vector<float> values(total);
        read_exact(input, values.data(), values.size(), "vector values");
        return DenseDataset(static_cast<std::uint32_t>(count),
                            static_cast<std::uint32_t>(dimension),
                            std::move(values));
    }

    const float *DenseDataset::row(std::uint32_t id) const
    {
        if (id >= count_)
            throw Error("vector row id is out of range");
        return values_.data() + static_cast<std::size_t>(id) * dimension_;
    }

    std::vector<Attribute> load_attributes(const std::string &path, std::uint32_t expected_count)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw Error("cannot open attribute file " + path);
        std::vector<Attribute> attributes(expected_count);
        read_exact(input, attributes.data(), attributes.size(), "attributes");
        return attributes;
    }

    std::vector<AttributeRange> load_ranges(const std::string &path, std::uint32_t expected_count)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw Error("cannot open query range file " + path);
        std::vector<AttributeRange> ranges(expected_count);
        for (auto &range : ranges)
        {
            read_exact(input, &range.low, 1, "range lower bound");
            read_exact(input, &range.high, 1, "range upper bound");
            if (range.low > range.high)
                std::swap(range.low, range.high);
        }
        return ranges;
    }

    std::vector<std::vector<ExternalId>> load_groundtruth(const std::string &path,
                                                          std::uint32_t expected_queries,
                                                          std::uint32_t expected_k)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw Error("cannot open groundtruth file " + path);
        std::int32_t count = 0;
        std::int32_t k = 0;
        read_exact(input, &count, 1, "groundtruth query count");
        read_exact(input, &k, 1, "groundtruth K");
        if (count < static_cast<std::int32_t>(expected_queries))
            throw Error("groundtruth contains fewer queries than requested");
        if (k != static_cast<std::int32_t>(expected_k))
            throw Error("groundtruth K does not match requested K");

        std::vector<std::vector<ExternalId>> result(expected_queries,
                                                    std::vector<ExternalId>(expected_k));
        for (auto &row : result)
            read_exact(input, row.data(), row.size(), "groundtruth row");
        return result;
    }

    void ensure_parent_directory(const std::string &path)
    {
        const std::filesystem::path parent = std::filesystem::path(path).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
    }

    void save_id_map(const std::string &path, const std::vector<ExternalId> &internal_to_external)
    {
        ensure_parent_directory(path);
        std::ofstream output(path, std::ios::binary);
        if (!output)
            throw Error("cannot open ID map for writing " + path);
        const char magic[8] = {'R', 'F', 'I', 'D', 'M', 'A', 'P', '\0'};
        const std::uint32_t version = 1;
        const std::uint32_t count = static_cast<std::uint32_t>(internal_to_external.size());
        output.write(magic, sizeof(magic));
        output.write(reinterpret_cast<const char *>(&version), sizeof(version));
        output.write(reinterpret_cast<const char *>(&count), sizeof(count));
        output.write(reinterpret_cast<const char *>(internal_to_external.data()),
                     static_cast<std::streamsize>(internal_to_external.size() * sizeof(ExternalId)));
        if (!output)
            throw Error("failed while writing ID map " + path);
    }

    std::vector<ExternalId> load_id_map(const std::string &path, std::uint32_t expected_count)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw Error("cannot open ID map " + path);
        char magic[8] = {};
        std::uint32_t version = 0;
        std::uint32_t count = 0;
        read_exact(input, magic, sizeof(magic), "ID map magic");
        read_exact(input, &version, 1, "ID map version");
        read_exact(input, &count, 1, "ID map count");
        const char expected_magic[8] = {'R', 'F', 'I', 'D', 'M', 'A', 'P', '\0'};
        if (std::memcmp(magic, expected_magic, sizeof(magic)) != 0)
            throw Error("ID map magic is invalid");
        if (version != 1)
            throw Error("ID map version is unsupported");
        if (count != expected_count)
            throw Error("ID map count does not match index point count");

        std::vector<ExternalId> result(count);
        read_exact(input, result.data(), result.size(), "ID map values");
        std::vector<ExternalId> sorted = result;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw Error("ID map contains duplicate external IDs");
        if (!sorted.empty() && sorted.back() == kInvalidPointId)
            throw Error("ID map contains the invalid ID sentinel");
        return result;
    }
}

