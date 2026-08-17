#pragma once

#include "rfanns/types.h"

#include <cstdint>
#include <vector>

namespace rfanns
{
    struct RangeTreeNode
    {
        std::int32_t node_id{-1};
        std::int32_t leaf_l{0};
        std::int32_t leaf_r{0};
        std::int32_t depth{0};
        std::int32_t parent{-1};
        std::int32_t left_child{-1};
        std::int32_t right_child{-1};
        std::vector<PointId> members;
    };

    class RangeTree
    {
    public:
        RangeTree() = default;
        RangeTree(std::uint32_t bins, double min_value, double max_value);

        void reset(std::uint32_t bins, double min_value, double max_value);
        std::uint16_t bin_for_value(double value) const;
        bool query_bins(double low, double high, std::uint16_t &bin_l, std::uint16_t &bin_r) const;

        void clear_members_and_graph_state();
        void reserve_members_from_bin_counts(const std::vector<std::uint32_t> &bin_counts);
        void add_member_to_path(std::uint16_t leaf, PointId id);
        void remove_member_from_path(std::uint16_t leaf, PointId id);
        void reconstruct_members(const std::vector<std::uint16_t> &leaf_ids,
                                 const std::vector<ExternalId> &internal_to_external,
                                 std::uint32_t point_count);

        std::vector<std::int32_t> canonical_cover(std::uint16_t query_l, std::uint16_t query_r) const;
        std::int32_t overlap(std::int32_t node_id, std::uint16_t query_l, std::uint16_t query_r) const;
        bool node_inside_query(std::int32_t node_id, std::uint16_t query_l, std::uint16_t query_r) const;
        bool node_contains_leaf(std::int32_t node_id, std::uint16_t leaf) const;

        std::uint32_t num_bins{0};
        std::uint32_t levels{0};
        std::uint32_t max_depth{0};
        double attr_min{0.0};
        double attr_max{0.0};
        double width{0.0};
        std::vector<RangeTreeNode> nodes;
        std::vector<std::vector<std::int32_t>> path_by_leaf;
        std::vector<std::int32_t> leaf_node_by_leaf;

    private:
        std::int32_t build_node(std::int32_t leaf_l,
                                std::int32_t leaf_r,
                                std::int32_t depth,
                                std::int32_t parent);
        void fill_paths(std::int32_t node_id, std::vector<std::int32_t> path);
        void canonical_cover_impl(std::int32_t node_id,
                                  std::uint16_t query_l,
                                  std::uint16_t query_r,
                                  std::vector<std::int32_t> &result) const;
    };
}
