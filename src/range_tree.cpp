#include "range_tree.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rfanns
{
    RangeTree::RangeTree(std::uint32_t bins, double min_value, double max_value)
    {
        reset(bins, min_value, max_value);
    }

    void RangeTree::reset(std::uint32_t bins, double min_value, double max_value)
    {
        if (bins == 0 || (bins & (bins - 1)) != 0)
            throw Error("num_bins must be a positive power of two");
        if (bins > static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::max()) + 1U)
            throw Error("num_bins must be at most 65536");
        if (!(max_value > min_value))
            throw Error("attr_max must be larger than attr_min");

        num_bins = bins;
        attr_min = min_value;
        attr_max = max_value;
        width = (attr_max - attr_min) / static_cast<double>(num_bins);
        max_depth = 0;
        for (std::uint32_t n = num_bins; n > 1; n >>= 1)
            ++max_depth;
        levels = max_depth + 1;

        nodes.clear();
        nodes.reserve(static_cast<std::size_t>(num_bins) * 2 - 1);
        path_by_leaf.assign(num_bins, std::vector<std::int32_t>(levels, -1));
        leaf_node_by_leaf.assign(num_bins, -1);
        build_node(0, static_cast<std::int32_t>(num_bins) - 1, 0, -1);
        fill_paths(0, {});
    }

    std::uint16_t RangeTree::bin_for_value(double value) const
    {
        if (value < attr_min || value > attr_max)
            throw Error("attribute value is outside the index domain");
        if (value == attr_max)
            return static_cast<std::uint16_t>(num_bins - 1);
        std::int64_t bin = static_cast<std::int64_t>(std::floor((value - attr_min) / width));
        bin = std::max<std::int64_t>(0, std::min<std::int64_t>(bin, num_bins - 1));
        return static_cast<std::uint16_t>(bin);
    }

    bool RangeTree::query_bins(double low,
                               double high,
                               std::uint16_t &bin_l,
                               std::uint16_t &bin_r) const
    {
        if (low > high)
            std::swap(low, high);
        if (high < attr_min || low > attr_max)
            return false;
        low = std::max(low, attr_min);
        high = std::min(high, attr_max);
        bin_l = bin_for_value(low);
        bin_r = bin_for_value(high);
        if (bin_l > bin_r)
            std::swap(bin_l, bin_r);
        return true;
    }

    void RangeTree::clear_members_and_graph_state()
    {
        for (auto &node : nodes)
            node.members.clear();
    }

    void RangeTree::reserve_members_from_bin_counts(const std::vector<std::uint32_t> &bin_counts)
    {
        if (bin_counts.size() != num_bins)
            throw Error("bin count size does not match num_bins");
        std::vector<std::uint64_t> prefix(num_bins + 1, 0);
        for (std::uint32_t bin = 0; bin < num_bins; ++bin)
            prefix[bin + 1] = prefix[bin] + bin_counts[bin];
        for (auto &node : nodes)
            node.members.reserve(static_cast<std::size_t>(prefix[node.leaf_r + 1] - prefix[node.leaf_l]));
    }

    void RangeTree::add_member_to_path(std::uint16_t leaf, PointId id)
    {
        if (leaf >= num_bins)
            throw Error("leaf id is outside num_bins");
        for (const std::int32_t node_id : path_by_leaf[leaf])
        {
            auto &node = nodes[static_cast<std::size_t>(node_id)];
            node.members.push_back(id);
        }
    }

    void RangeTree::remove_member_from_path(std::uint16_t leaf, PointId id)
    {
        if (leaf >= num_bins)
            throw Error("leaf id is outside num_bins");
        for (const std::int32_t node_id : path_by_leaf[leaf])
        {
            auto &members = nodes[static_cast<std::size_t>(node_id)].members;
            const auto it = std::find(members.begin(), members.end(), id);
            if (it != members.end())
                members.erase(it);
        }
    }

    void RangeTree::reconstruct_members(const std::vector<std::uint16_t> &leaf_ids,
                                        const std::vector<ExternalId> &internal_to_external,
                                        std::uint32_t point_count)
    {
        if (point_count > leaf_ids.size() ||
            (!internal_to_external.empty() && point_count > internal_to_external.size()))
            throw Error("point metadata is too small while reconstructing tree members");
        std::vector<std::uint32_t> counts(num_bins, 0);
        for (PointId id = 0; id < point_count; ++id)
        {
            if (!internal_to_external.empty() &&
                internal_to_external[id] == kInvalidPointId)
                continue;
            if (leaf_ids[id] >= num_bins)
                throw Error("leaf id is outside num_bins while reconstructing tree members");
            ++counts[leaf_ids[id]];
        }
        clear_members_and_graph_state();
        reserve_members_from_bin_counts(counts);
        for (PointId id = 0; id < point_count; ++id)
        {
            if (internal_to_external.empty() ||
                internal_to_external[id] != kInvalidPointId)
                add_member_to_path(leaf_ids[id], id);
        }
    }

    std::vector<std::int32_t> RangeTree::canonical_cover(std::uint16_t query_l,
                                                         std::uint16_t query_r) const
    {
        if (query_l > query_r)
            return {};
        std::vector<std::int32_t> result;
        canonical_cover_impl(0, query_l, query_r, result);
        return result;
    }

    std::int32_t RangeTree::overlap(std::int32_t node_id,
                                    std::uint16_t query_l,
                                    std::uint16_t query_r) const
    {
        const auto &node = nodes.at(static_cast<std::size_t>(node_id));
        const std::int32_t left = std::max(node.leaf_l, static_cast<std::int32_t>(query_l));
        const std::int32_t right = std::min(node.leaf_r, static_cast<std::int32_t>(query_r));
        return std::max<std::int32_t>(0, right - left + 1);
    }

    bool RangeTree::node_inside_query(std::int32_t node_id,
                                      std::uint16_t query_l,
                                      std::uint16_t query_r) const
    {
        const auto &node = nodes.at(static_cast<std::size_t>(node_id));
        return node.leaf_l >= query_l && node.leaf_r <= query_r;
    }

    bool RangeTree::node_contains_leaf(std::int32_t node_id, std::uint16_t leaf) const
    {
        const auto &node = nodes.at(static_cast<std::size_t>(node_id));
        return node.leaf_l <= leaf && leaf <= node.leaf_r;
    }

    std::int32_t RangeTree::build_node(std::int32_t leaf_l,
                                       std::int32_t leaf_r,
                                       std::int32_t depth,
                                       std::int32_t parent)
    {
        const std::int32_t node_id = static_cast<std::int32_t>(nodes.size());
        nodes.emplace_back();
        auto &node = nodes.back();
        node.node_id = node_id;
        node.leaf_l = leaf_l;
        node.leaf_r = leaf_r;
        node.depth = depth;
        node.parent = parent;
        if (leaf_l == leaf_r)
        {
            leaf_node_by_leaf[leaf_l] = node_id;
            return node_id;
        }
        const std::int32_t mid = (leaf_l + leaf_r) / 2;
        const std::int32_t left = build_node(leaf_l, mid, depth + 1, node_id);
        const std::int32_t right = build_node(mid + 1, leaf_r, depth + 1, node_id);
        nodes[node_id].left_child = left;
        nodes[node_id].right_child = right;
        return node_id;
    }

    void RangeTree::fill_paths(std::int32_t node_id, std::vector<std::int32_t> path)
    {
        path.push_back(node_id);
        const auto &node = nodes[node_id];
        if (node.left_child < 0 && node.right_child < 0)
        {
            if (path.size() != levels)
                throw Error("leaf depth does not match RangeTree levels");
            path_by_leaf[node.leaf_l] = std::move(path);
            return;
        }
        fill_paths(node.left_child, path);
        fill_paths(node.right_child, std::move(path));
    }

    void RangeTree::canonical_cover_impl(std::int32_t node_id,
                                         std::uint16_t query_l,
                                         std::uint16_t query_r,
                                         std::vector<std::int32_t> &result) const
    {
        const auto &node = nodes[node_id];
        if (node.leaf_r < query_l || node.leaf_l > query_r)
            return;
        if (node.leaf_l >= query_l && node.leaf_r <= query_r)
        {
            result.push_back(node_id);
            return;
        }
        if (node.left_child >= 0)
            canonical_cover_impl(node.left_child, query_l, query_r, result);
        if (node.right_child >= 0)
            canonical_cover_impl(node.right_child, query_l, query_r, result);
    }
}
