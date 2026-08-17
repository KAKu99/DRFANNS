#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rfanns
{
    using PointId = std::uint32_t;
    using ExternalId = std::uint32_t;
    using Attribute = std::int32_t;

    constexpr PointId kInvalidPointId = std::numeric_limits<PointId>::max();

    class Error : public std::runtime_error
    {
    public:
        explicit Error(const std::string &message) : std::runtime_error(message) {}
    };

    struct AttributeRange
    {
        Attribute low{0};
        Attribute high{0};
    };

    enum class DomainMode
    {
        Fixed,
        Derived
    };

    struct DomainOptions
    {
        DomainMode mode{DomainMode::Fixed};
        double attr_min{0.0};
        double attr_max{0.0};
        std::uint32_t num_bins{2048};
        double min_segment_width{0.0};
    };

    struct BuildOptions
    {
        std::uint32_t M{32};
        std::uint32_t ef_construction{200};
        std::uint32_t threads{1};
        std::uint64_t seed{0};
        DomainOptions domain;
    };

    struct SearchOptions
    {
        std::uint32_t k{10};
        std::uint32_t ef{100};
    };

    struct InsertOptions
    {
        bool dynamic_attribute_expansion{false};
    };

    struct SearchResult
    {
        ExternalId id{kInvalidPointId};
        float distance{0.0f};
    };

    struct IndexMetadata
    {
        std::uint32_t num_points{0};
        std::uint32_t active_points{0};
        std::uint32_t dimension{0};
        std::uint32_t M{0};
        std::uint32_t ef_construction{0};
        std::uint32_t num_bins{0};
        std::uint32_t levels{0};
        double attr_min{0.0};
        double attr_max{0.0};
    };

    struct UpdateStats
    {
        std::uint64_t inserted_points{0};
        std::uint64_t deleted_points{0};
        std::uint64_t delete_segments_processed{0};
        std::uint64_t delete_search_candidates{0};
        std::uint64_t delete_incoming_owners{0};
        std::uint64_t delete_outgoing_edges_removed{0};
        std::uint64_t delete_incoming_edges_removed{0};
        std::uint64_t delete_replacement_edges_added{0};
        std::uint64_t delete_edge_lists_rewritten{0};
        std::uint64_t delete_remaining_edges{0};
        std::uint64_t forward_lists_rewritten{0};
        std::uint64_t reverse_adjacency_edges{0};
        std::uint64_t reverse_adjacency_updates{0};
        std::uint64_t attr_expand_count{0};
        std::uint64_t attr_expand_left_count{0};
        std::uint64_t attr_expand_right_count{0};
        double delete_seconds{0.0};
        double reverse_adjacency_build_seconds{0.0};
        double reverse_adjacency_update_seconds{0.0};
        double forward_list_rewrite_seconds{0.0};
        double attr_expand_seconds{0.0};
    };
}
