#pragma once

#include "rfanns/index.h"
#include "range_tree.h"
#include "hnswlib/hnswlib.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rfanns
{
    inline constexpr std::array<char, 8> kIndexFileMagic = {
        'R', 'S', 'H', 'N', 'S', 'W', 'E', '\0'};

    using EdgeRecord = std::uint64_t;
    using AttrKey = std::uint32_t;

    inline EdgeRecord make_edge_record(PointId destination, AttrKey attribute) noexcept
    {
        return (static_cast<EdgeRecord>(attribute) << 32) | destination;
    }

    inline PointId edge_destination(EdgeRecord record) noexcept
    {
        return static_cast<PointId>(record);
    }

    inline AttrKey edge_attribute(EdgeRecord record) noexcept
    {
        return static_cast<AttrKey>(record >> 32);
    }

    inline AttrKey attribute_key(Attribute value) noexcept
    {
        return static_cast<AttrKey>(value) ^ 0x80000000U;
    }

    class AlignedFloatBuffer
    {
    public:
        AlignedFloatBuffer() = default;
        ~AlignedFloatBuffer();
        AlignedFloatBuffer(const AlignedFloatBuffer &) = delete;
        AlignedFloatBuffer &operator=(const AlignedFloatBuffer &) = delete;

        void allocate(std::size_t floats);
        void grow(std::size_t old_floats, std::size_t new_floats);
        float *data() noexcept { return data_; }
        const float *data() const noexcept { return data_; }
        bool empty() const noexcept { return data_ == nullptr; }

    private:
        float *data_{nullptr};
        std::size_t allocated_floats_{0};
    };

    struct ThreadScratch
    {
        explicit ThreadScratch(std::size_t size = 0) : visited(size, 0) {}
        std::uint32_t next_tag();

        std::vector<std::uint32_t> visited;
        std::uint32_t tag{1};
        // Reused by construction search.  Keeping the heap storage per worker
        // avoids growing and freeing two priority-queue vectors for every
        // point at every segment-tree depth.
        std::vector<std::pair<float, PointId>> candidate_heap;
        std::vector<std::pair<float, PointId>> top_heap;
        std::vector<std::pair<float, PointId>> candidate_distances;
        std::vector<PointId> selected;
        std::vector<PointId> adjacency_snapshot;
    };

    struct DeleteLocalStats
    {
        std::uint64_t segments{0};
        std::uint64_t search_candidates{0};
        std::uint64_t incoming_owners{0};
        std::uint64_t outgoing_removed{0};
        std::uint64_t incoming_removed{0};
        std::uint64_t replacements_added{0};
        std::uint64_t lists_rewritten{0};
        std::uint64_t remaining_edges{0};
        std::int64_t reverse_edge_count_change{0};
        std::uint64_t reverse_updates{0};
        std::uint64_t forward_lists_rewritten{0};
        double neighbor_rewrite_seconds{0.0};
    };

    struct NeighborRewriteSummary
    {
        std::int64_t reverse_edge_count_change{0};
        std::uint64_t reverse_updates{0};
        std::uint64_t forward_lists_rewritten{0};

        void add(const NeighborRewriteSummary &other) noexcept
        {
            reverse_edge_count_change += other.reverse_edge_count_change;
            reverse_updates += other.reverse_updates;
            forward_lists_rewritten += other.forward_lists_rewritten;
        }
    };

    class IndexImpl
    {
    public:
        explicit IndexImpl(std::shared_ptr<const DenseDataset> data, std::uint64_t random_seed);

        using DistanceId = std::pair<float, PointId>;

        std::shared_ptr<const DenseDataset> dataset;
        RangeTree tree;
        std::vector<Attribute> attributes;
        std::vector<AttrKey> attribute_keys;
        std::vector<std::uint16_t> leaf_ids;
        std::vector<std::uint16_t> degree;
        std::vector<std::uint32_t> degree_by_depth;
        std::vector<std::size_t> depth_offsets;
        std::uint32_t capacity_per_point{0};

        // Sole forward-adjacency storage: destination in the low 32 bits and
        // its order-preserving attribute key in the high 32 bits.
        std::vector<EdgeRecord> edge_records;
        std::vector<AttrKey> edge_scope_low;
        std::vector<AttrKey> edge_scope_high;

        std::vector<std::vector<PointId>> reverse_adjacency;
        bool reverse_adjacency_ready{false};

        std::vector<ExternalId> internal_to_external;
        std::vector<PointId> external_to_internal;
        std::vector<PointId> reusable_ids;

        AlignedFloatBuffer internal_vectors;
        std::size_t vector_stride{0};
        std::vector<std::uint32_t> visited;
        std::uint32_t visited_tag{1};

        std::vector<std::uint32_t> cut_token_by_leaf;
        std::vector<std::uint8_t> cut_count_by_leaf;
        std::vector<std::uint8_t> cut_depths_by_leaf;
        std::uint32_t cut_token{1};

        std::uint32_t num_points{0};
        std::uint32_t active_points{0};
        std::uint32_t point_capacity{0};
        std::uint32_t dimension{0};
        std::uint32_t M{0};
        std::uint32_t ef_construction{0};
        std::uint64_t seed{0};
        UpdateStats stats;

        std::unique_ptr<hnswlib::L2Space> space;
        hnswlib::DISTFUNC<float> distance_function{nullptr};
        void *distance_parameter{nullptr};

        bool search_prepared() const noexcept;
        void initialize_distance();
        void initialize_layout();
        inline std::uint32_t capacity_at_depth(std::uint32_t depth) const
        {
            return degree_by_depth[depth];
        }

        inline std::size_t degree_index(PointId id, std::uint32_t depth) const
        {
            return static_cast<std::size_t>(id) * tree.levels + depth;
        }

        inline std::size_t edge_list_index(PointId id, std::uint32_t depth) const
        {
            return degree_index(id, depth);
        }

        inline std::size_t neighbor_index(PointId id,
                                          std::uint32_t depth,
                                          std::uint32_t slot) const
        {
            return static_cast<std::size_t>(id) * capacity_per_point + depth_offsets[depth] + slot;
        }

        inline std::size_t edge_list_capacity_count() const
        {
            return static_cast<std::size_t>(point_capacity) * tree.levels;
        }

        inline std::uint16_t get_degree(PointId id, std::uint32_t depth) const
        {
            return degree[degree_index(id, depth)];
        }

        inline PointId get_neighbor(PointId id,
                                    std::uint32_t depth,
                                    std::uint32_t slot) const
        {
            return edge_destination(edge_records[neighbor_index(id, depth, slot)]);
        }

        inline const float *point_data(PointId id) const
        {
            if (!internal_vectors.empty())
                return internal_vectors.data() + static_cast<std::size_t>(id) * vector_stride;
            const ExternalId external = internal_to_external.empty() ? id : internal_to_external[id];
            return dataset->row(external);
        }

        inline float distance(PointId left, PointId right) const
        {
            return distance_function(point_data(left), point_data(right), distance_parameter);
        }

        inline float distance_to_query(const float *query, PointId id) const
        {
            return distance_function(query, point_data(id), distance_parameter);
        }

        void materialize_identity_id_map();
        void rebuild_external_map();
        bool slot_is_bound(PointId id) const noexcept;
        PointId to_internal(ExternalId id) const;
        ExternalId to_external(PointId id) const;

        void prepare_search();
        void prepare_updates();
        void reserve(std::uint32_t capacity);
        void build_internal_vectors();
        void build_edge_metadata();
        void build_reverse_adjacency(std::uint32_t threads = 0);
        void rebuild_edge_scope(PointId id, std::uint32_t depth);
        NeighborRewriteSummary write_neighbor_list(PointId owner,
                                                   std::uint32_t depth,
                                                   const std::vector<PointId> &list);
        void commit_neighbor_rewrite_stats(const NeighborRewriteSummary &summary,
                                           double batch_seconds);
        std::vector<PointId> current_neighbors(PointId id, std::uint32_t depth) const;
        std::vector<PointId> select_hnsw_neighbors(PointId center,
                                                   const std::vector<PointId> &candidates,
                                                   std::uint32_t limit) const;
        std::vector<PointId> select_hnsw_neighbors_scored(
            PointId center,
            std::vector<DistanceId> &candidates,
            std::uint32_t limit) const;
        std::vector<PointId> select_hnsw_neighbors_scored(
            PointId center,
            const std::vector<DistanceId> &candidates,
            std::uint32_t limit) const;

        std::uint32_t next_visited_tag();
        void begin_cut_cache_query();
        const std::uint8_t *cut_depths(std::uint16_t leaf,
                                      std::uint16_t query_l,
                                      std::uint16_t query_r,
                                      std::uint8_t &count);

        std::vector<SearchResult> search(const float *query,
                                         std::uint32_t query_id,
                                         AttributeRange range,
                                         const SearchOptions &options);
        void append_query_seed(std::int32_t node_id,
                               std::uint32_t query_id,
                               AttributeRange range,
                               std::vector<PointId> &seeds) const;
        void select_edges(PointId current,
                          std::uint16_t query_l,
                          std::uint16_t query_r,
                          AttrKey attr_l,
                          AttrKey attr_r,
                          std::uint32_t edge_limit,
                          std::uint32_t visit_tag,
                          std::vector<PointId> &selected);

        PointId insert(ExternalId external_id,
                       const float *vector,
                       Attribute attribute,
                       const InsertOptions &options);
        void expand_domain_to_include(Attribute attribute);
        void expand_domain_once(double new_min,
                                double new_max,
                                std::uint32_t old_leaf_offset,
                                bool left_expansion);
        void search_insert_candidates(std::int32_t node_id,
                                      PointId query_id,
                                      std::uint32_t depth,
                                      PointId preferred_seed,
                                      std::int32_t inherited_node_id,
                                      const std::vector<DistanceId> &inherited,
                                      std::vector<DistanceId> &result_distances);
        void insert_and_prune(PointId owner,
                              std::uint32_t depth,
                              PointId candidate,
                              NeighborRewriteSummary &rewrite_summary);
        void build_insert_edges(PointId id, NeighborRewriteSummary &rewrite_summary);

        bool erase(ExternalId external_id);
        bool erase_sequential(PointId id);
        std::vector<std::vector<PointId>> collect_outgoing(PointId deleted) const;
        std::vector<PointId> repair_search(PointId deleted,
                                           std::uint32_t depth);
        std::vector<PointId> incoming_owners(PointId deleted,
                                             std::uint32_t depth) const;
        void repair_delete_depth(PointId deleted,
                                 std::uint32_t depth,
                                 const std::vector<PointId> &outgoing,
                                 const std::vector<PointId> &owners,
                                 const std::vector<PointId> &search_candidates,
                                 DeleteLocalStats &local);
        void finalize_delete(PointId id);
        std::vector<PointId> select_repaired_list(PointId owner,
                                                  std::uint32_t depth,
                                                  const std::vector<PointId> &survivors,
                                                  const std::vector<PointId> &forced,
                                                  const std::vector<PointId> &pool,
                                                  std::uint32_t target_degree) const;
        void save(const std::string &path) const;
        static std::unique_ptr<IndexImpl> load(const std::string &path,
                                               std::shared_ptr<const DenseDataset> dataset,
                                               std::uint64_t seed,
                                               bool allow_data_superset);
        void validate() const;
    };

    std::uint64_t splitmix64(std::uint64_t value) noexcept;
}
