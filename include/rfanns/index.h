#pragma once

#include "rfanns/dataset.h"
#include "rfanns/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rfanns
{
    class IndexImpl;
    namespace testing
    {
        class IndexAccess;
    }

    class Index
    {
    public:
        Index(Index &&) noexcept;
        Index &operator=(Index &&) noexcept;
        ~Index();

        Index(const Index &) = delete;
        Index &operator=(const Index &) = delete;

        static Index load(const std::string &index_path,
                          std::shared_ptr<const DenseDataset> dataset,
                          const std::string &id_map_path = {},
                          bool allow_data_superset = false,
                          std::uint64_t seed = 0);

        void prepare_search();
        void prepare_updates();
        void reserve(std::uint32_t point_capacity);

        std::vector<SearchResult> search(const float *query,
                                         std::uint32_t query_id,
                                         AttributeRange range,
                                         const SearchOptions &options);
        PointId insert(ExternalId external_id,
                       const float *vector,
                       Attribute attribute,
                       const InsertOptions &options = {});
        bool erase(ExternalId external_id);

        void save(const std::string &index_path, const std::string &id_map_path = {}) const;
        void validate() const;

        IndexMetadata metadata() const;
        const UpdateStats &update_stats() const;
        std::vector<ExternalId> current_id_map() const;

    private:
        explicit Index(std::unique_ptr<IndexImpl> impl);
        std::unique_ptr<IndexImpl> impl_;

        friend class IndexBuilder;
        friend class testing::IndexAccess;
    };

    class IndexBuilder
    {
    public:
        static Index build(std::shared_ptr<const DenseDataset> dataset,
                           const std::vector<Attribute> &attributes,
                           const BuildOptions &options);
    };
}
