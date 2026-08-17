#include "index_internal.h"

#include <algorithm>
#include <limits>

namespace rfanns
{
    namespace
    {
        constexpr std::uint32_t kSelectEdgePrefetchLines = 4;
        constexpr std::uint32_t kDistancePrefetchCount = 3;

        void prefetch_cache_lines(const void *address, std::size_t bytes, std::uint32_t maximum_lines)
        {
            const auto *base = static_cast<const char *>(address);
            const std::uint32_t lines = static_cast<std::uint32_t>(
                std::min<std::size_t>(maximum_lines, (bytes + 63) / 64));
            for (std::uint32_t line = 0; line < lines; ++line)
                __builtin_prefetch(base + static_cast<std::size_t>(line) * 64, 0, 3);
        }
    }

    void IndexImpl::append_query_seed(std::int32_t node_id,
                                      std::uint32_t query_id,
                                      AttributeRange range,
                                      std::vector<PointId> &seeds) const
    {
        const auto &members = tree.nodes.at(static_cast<std::size_t>(node_id)).members;
        if (members.empty())
            return;
        const std::uint64_t state = splitmix64(static_cast<std::uint64_t>(node_id) ^
                                               (static_cast<std::uint64_t>(query_id) << 32) ^
                                               seed ^ 0x9e3779b97f4a7c15ULL);
        const std::size_t start = static_cast<std::size_t>(state % members.size());
        for (std::size_t offset = 0; offset < members.size(); ++offset)
        {
            const PointId id = members[(start + offset) % members.size()];
            if (attributes[id] < range.low || attributes[id] > range.high)
                continue;
            seeds.push_back(id);
            return;
        }
    }

    void IndexImpl::select_edges(PointId current,
                                 std::uint16_t query_l,
                                 std::uint16_t query_r,
                                 AttrKey attr_l,
                                 AttrKey attr_r,
                                 std::uint32_t edge_limit,
                                 std::uint32_t visit_tag,
                                 std::vector<PointId> &selected)
    {
        selected.clear();
        std::uint8_t depth_count = 0;
        const std::uint8_t *depth_values = cut_depths(leaf_ids[current],
                                                      query_l,
                                                      query_r,
                                                      depth_count);
        for (std::uint8_t depth_index = 0; depth_index < depth_count; ++depth_index)
        {
            const std::uint32_t depth = depth_values[depth_index];
            if (depth_index + 1 < depth_count)
            {
                const std::uint32_t next_depth = depth_values[depth_index + 1];
                const std::size_t next_base = neighbor_index(current, next_depth, 0);
                prefetch_cache_lines(edge_records.data() + next_base,
                                     static_cast<std::size_t>(capacity_at_depth(next_depth)) * sizeof(EdgeRecord),
                                     kSelectEdgePrefetchLines);
            }

            const std::size_t base = neighbor_index(current, depth, 0);
            const std::uint16_t count = get_degree(current, depth);
            const std::size_t list_index = edge_list_index(current, depth);
            const bool scope_contained = count != 0 &&
                                         edge_scope_low[list_index] >= attr_l &&
                                         edge_scope_high[list_index] <= attr_r;
            for (std::uint16_t slot = 0; slot < count; ++slot)
            {
                const EdgeRecord record = edge_records[base + slot];
                if (!scope_contained)
                {
                    const AttrKey neighbor_attribute = edge_attribute(record);
                    if (neighbor_attribute < attr_l || neighbor_attribute > attr_r)
                        continue;
                }
                const PointId neighbor = edge_destination(record);
                if (visited[neighbor] == visit_tag)
                    continue;
                visited[neighbor] = visit_tag;
                selected.push_back(neighbor);
                if (selected.size() >= edge_limit)
                    return;
            }
        }
    }

    std::vector<SearchResult> IndexImpl::search(const float *query,
                                                std::uint32_t query_id,
                                                AttributeRange range,
                                                const SearchOptions &options)
    {
        if (query == nullptr)
            throw Error("search query pointer is null");
        if (options.k == 0 || options.ef == 0)
            throw Error("search K and ef must be positive");
        prepare_search();
        if (range.low > range.high)
            std::swap(range.low, range.high);

        std::uint16_t query_l = 0;
        std::uint16_t query_r = 0;
        if (!tree.query_bins(range.low, range.high, query_l, query_r))
            return {};
        const AttrKey attr_l = attribute_key(range.low);
        const AttrKey attr_r = attribute_key(range.high);
        const std::uint32_t edge_limit = M;

        begin_cut_cache_query();
        const std::vector<std::int32_t> cover = tree.canonical_cover(query_l, query_r);
        const std::uint32_t tag = next_visited_tag();
        std::priority_queue<DistanceId, std::vector<DistanceId>, std::greater<DistanceId>> candidates;
        std::priority_queue<DistanceId> top;
        std::vector<PointId> seeds;
        seeds.reserve(cover.size());
        for (const std::int32_t node_id : cover)
            append_query_seed(node_id, query_id, range, seeds);

        for (PointId seed : seeds)
        {
            if (visited[seed] == tag)
                continue;
            visited[seed] = tag;
            const float value = distance_to_query(query, seed);
            candidates.emplace(value, seed);
            top.emplace(value, seed);
            if (top.size() > options.ef)
                top.pop();
        }
        if (top.empty())
            return {};

        float lower_bound = top.top().first;
        std::vector<PointId> selected;
        selected.reserve(edge_limit);
        while (!candidates.empty())
        {
            const DistanceId current = candidates.top();
            if (top.size() >= options.ef && current.first > lower_bound)
                break;
            candidates.pop();

            select_edges(current.second,
                         query_l,
                         query_r,
                         attr_l,
                         attr_r,
                         edge_limit,
                         tag,
                         selected);
            for (std::size_t index = 0;
                 index < std::min<std::size_t>(kDistancePrefetchCount, selected.size());
                 ++index)
            {
                prefetch_cache_lines(point_data(selected[index]),
                                     static_cast<std::size_t>(dimension) * sizeof(float),
                                     28);
            }

            for (PointId neighbor : selected)
            {
                const float value = distance_to_query(query, neighbor);
                if (top.size() < options.ef || value < lower_bound)
                {
                    candidates.emplace(value, neighbor);
                    top.emplace(value, neighbor);
                    if (top.size() > options.ef)
                        top.pop();
                    lower_bound = top.top().first;
                }
            }
        }

        while (top.size() > options.k)
            top.pop();
        std::vector<SearchResult> result;
        result.reserve(top.size());
        while (!top.empty())
        {
            const auto item = top.top();
            top.pop();
            result.push_back({to_external(item.second), item.first});
        }
        std::sort(result.begin(), result.end(), [](const SearchResult &left, const SearchResult &right) {
            return left.distance == right.distance ? left.id < right.id : left.distance < right.distance;
        });
        return result;
    }
}
