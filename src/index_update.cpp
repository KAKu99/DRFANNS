#include "index_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rfanns
{
    void IndexImpl::search_insert_candidates(std::int32_t node_id,
                                             PointId query_id,
                                             std::uint32_t depth,
                                             PointId preferred_seed,
                                             std::int32_t inherited_node_id,
                                             const std::vector<DistanceId> &inherited,
                                             std::vector<DistanceId> &result_distances)
    {
        result_distances.clear();
        if (node_id < 0 || static_cast<std::size_t>(node_id) >= tree.nodes.size())
            return;
        const auto &node = tree.nodes[node_id];
        PointId seed = preferred_seed;
        if (seed == kInvalidPointId)
        {
            for (PointId member : node.members)
            {
                if (member != query_id)
                {
                    seed = member;
                    break;
                }
            }
        }
        if (seed == kInvalidPointId)
            return;

        const std::uint32_t search_ef =
            std::max(capacity_at_depth(depth), ef_construction == 0 ? capacity_at_depth(depth)
                                                                    : ef_construction);
        const std::uint32_t tag = next_visited_tag();
        const bool skip_inherited_region = inherited_node_id >= 0 && !inherited.empty();
        const std::int32_t inherited_region_leaf_l = skip_inherited_region
                                                  ? tree.nodes[inherited_node_id].leaf_l
                                                  : 0;
        const std::int32_t inherited_region_leaf_r = skip_inherited_region
                                                  ? tree.nodes[inherited_node_id].leaf_r
                                                  : -1;
        std::priority_queue<DistanceId, std::vector<DistanceId>, std::greater<DistanceId>> candidates;
        std::priority_queue<DistanceId> top;
        bool seed_inherited = false;
        float seed_distance = 0.0f;
        for (const DistanceId &candidate : inherited)
        {
            const PointId inherited_id = candidate.second;
            if (visited[inherited_id] == tag)
                continue;
            visited[inherited_id] = tag;
            top.push(candidate);
            candidates.push(candidate);
            if (top.size() > search_ef)
                top.pop();
            if (inherited_id == seed)
            {
                seed_inherited = true;
                seed_distance = candidate.first;
            }
        }
        if (!seed_inherited)
        {
            seed_distance = distance(query_id, seed);
            visited[seed] = tag;
            top.emplace(seed_distance, seed);
            if (top.size() > search_ef)
                top.pop();
        }
        if (!seed_inherited)
            candidates.emplace(seed_distance, seed);
        float lower_bound = top.top().first;

        while (!candidates.empty())
        {
            const DistanceId current = candidates.top();
            candidates.pop();
            if (top.size() >= search_ef && current.first > lower_bound)
                break;
            for (std::uint16_t slot = 0; slot < get_degree(current.second, depth); ++slot)
            {
                const PointId neighbor = get_neighbor(current.second, depth, slot);
                if (visited[neighbor] == tag)
                    continue;
                visited[neighbor] = tag;
                const std::int32_t neighbor_leaf = leaf_ids[neighbor];
                if (skip_inherited_region && neighbor_leaf >= inherited_region_leaf_l &&
                    neighbor_leaf <= inherited_region_leaf_r)
                    continue;
                const float value = distance(query_id, neighbor);
                if (top.size() < search_ef || value < lower_bound)
                {
                    candidates.emplace(value, neighbor);
                    top.emplace(value, neighbor);
                    if (top.size() > search_ef)
                        top.pop();
                    lower_bound = top.top().first;
                }
            }
        }
        result_distances.reserve(top.size());
        while (!top.empty())
        {
            const DistanceId candidate = top.top();
            if (candidate.second != query_id)
                result_distances.push_back(candidate);
            top.pop();
        }
    }

    void IndexImpl::insert_and_prune(PointId owner,
                                     std::uint32_t depth,
                                     PointId candidate,
                                     NeighborRewriteSummary &rewrite_summary)
    {
        std::vector<PointId> pool = current_neighbors(owner, depth);
        if (std::find(pool.begin(), pool.end(), candidate) != pool.end())
            return;
        pool.push_back(candidate);
        if (pool.size() > capacity_at_depth(depth))
            pool = select_hnsw_neighbors(owner, pool, capacity_at_depth(depth));
        rewrite_summary.add(write_neighbor_list(owner, depth, pool));
    }

    void IndexImpl::build_insert_edges(PointId id,
                                       NeighborRewriteSummary &rewrite_summary)
    {
        const auto &path = tree.path_by_leaf.at(leaf_ids[id]);
        std::vector<DistanceId> parent_candidates;
        std::vector<DistanceId> candidate_distances;
        PointId parent_seed = kInvalidPointId;
        std::int32_t inherited_node_id = -1;
        for (std::uint32_t remaining = tree.levels; remaining > 0; --remaining)
        {
            const std::uint32_t depth = remaining - 1;
            const std::int32_t node_id = path[depth];
            const auto &node = tree.nodes.at(static_cast<std::size_t>(node_id));
            if (node.members.size() == 1)
            {
                rewrite_summary.add(write_neighbor_list(id, depth, {}));
                parent_seed = kInvalidPointId;
                inherited_node_id = -1;
                parent_candidates.clear();
                continue;
            }
            search_insert_candidates(node_id,
                                     id,
                                     depth,
                                     parent_seed,
                                     inherited_node_id,
                                     parent_candidates,
                                     candidate_distances);

            parent_seed = kInvalidPointId;
            float best_distance = std::numeric_limits<float>::infinity();
            for (const DistanceId &candidate : candidate_distances)
            {
                if (candidate.first < best_distance ||
                    (candidate.first == best_distance && candidate.second < parent_seed))
                {
                    best_distance = candidate.first;
                    parent_seed = candidate.second;
                }
            }

            const std::vector<PointId> selected =
                select_hnsw_neighbors_scored(id,
                                             candidate_distances,
                                             capacity_at_depth(depth));
            rewrite_summary.add(write_neighbor_list(id, depth, selected));
            for (PointId neighbor : selected)
                insert_and_prune(neighbor, depth, id, rewrite_summary);
            parent_candidates.swap(candidate_distances);
            inherited_node_id = node_id;
        }
    }

    PointId IndexImpl::insert(ExternalId external_id,
                              const float *vector,
                              Attribute attribute,
                              const InsertOptions &options)
    {
        if (vector == nullptr)
            throw Error("insert vector pointer is null");
        prepare_updates();
        if (to_internal(external_id) != kInvalidPointId)
            throw Error("insert external ID is already active");
        if (options.dynamic_attribute_expansion)
            expand_domain_to_include(attribute);
        else if (attribute < tree.attr_min || attribute > tree.attr_max)
            throw Error("insert attribute is outside the index domain and dynamic expansion is disabled");

        const bool reuse = !reusable_ids.empty();
        if (!reuse && num_points >= point_capacity)
            throw Error("point capacity exceeded; call Index::reserve before inserting");
        PointId id = kInvalidPointId;
        if (reuse)
        {
            id = reusable_ids.back();
            reusable_ids.pop_back();
            if (id >= num_points || slot_is_bound(id))
                throw Error("reusable internal ID state is invalid");
        }
        else
        {
            id = num_points++;
        }

        if (internal_to_external.empty() && external_id != id)
            materialize_identity_id_map();
        if (!internal_to_external.empty())
        {
            if (internal_to_external.size() < point_capacity)
                internal_to_external.resize(point_capacity, kInvalidPointId);
            const ExternalId previous = internal_to_external[id];
            if (previous < external_to_internal.size() && external_to_internal[previous] == id)
                external_to_internal[previous] = kInvalidPointId;
            internal_to_external[id] = external_id;
            if (external_to_internal.size() <= external_id)
                external_to_internal.resize(static_cast<std::size_t>(external_id) + 1, kInvalidPointId);
            external_to_internal[external_id] = id;
        }

        float *destination = internal_vectors.data() + static_cast<std::size_t>(id) * vector_stride;
        std::copy_n(vector, dimension, destination);
        std::fill(destination + dimension, destination + vector_stride, 0.0f);
        attributes[id] = attribute;
        attribute_keys[id] = attribute_key(attribute);
        ++active_points;
        ++stats.inserted_points;

        NeighborRewriteSummary rewrite_summary;
        const auto rewrite_started = std::chrono::steady_clock::now();
        for (std::uint32_t depth = 0; depth < tree.levels; ++depth)
            rewrite_summary.add(write_neighbor_list(id, depth, {}));
        leaf_ids[id] = tree.bin_for_value(attribute);
        tree.add_member_to_path(leaf_ids[id], id);
        build_insert_edges(id, rewrite_summary);
        commit_neighbor_rewrite_stats(
            rewrite_summary,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          rewrite_started)
                .count());
        return id;
    }

    void IndexImpl::expand_domain_to_include(Attribute attribute)
    {
        while (attribute < tree.attr_min || attribute > tree.attr_max)
        {
            if (tree.num_bins > 32768)
                throw Error("dynamic attribute expansion would exceed 65536 bins");
            const double old_min = tree.attr_min;
            const double old_max = tree.attr_max;
            const double domain_width = old_max - old_min;
            if (!(domain_width > 0.0))
                throw Error("dynamic expansion requires a positive domain width");
            if (attribute < old_min)
                expand_domain_once(old_min - domain_width,
                                   old_max,
                                   tree.num_bins,
                                   true);
            else
                expand_domain_once(old_min,
                                   old_max + domain_width,
                                   0,
                                   false);
        }
    }

    void IndexImpl::expand_domain_once(double new_min,
                                       double new_max,
                                       std::uint32_t old_leaf_offset,
                                       bool left_expansion)
    {
        const auto started = std::chrono::steady_clock::now();
        const std::uint32_t old_bins = tree.num_bins;
        const std::uint32_t old_levels = tree.levels;
        const std::uint32_t old_capacity_per_point = capacity_per_point;
        const std::vector<std::uint32_t> old_degree_by_depth = degree_by_depth;
        const std::vector<std::size_t> old_depth_offsets = depth_offsets;
        std::vector<std::uint16_t> old_degree = std::move(degree);
        std::vector<EdgeRecord> old_edge_records = std::move(edge_records);
        const bool had_reverse = reverse_adjacency_ready;

        tree.reset(old_bins * 2, new_min, new_max);
        degree_by_depth.clear();
        degree_by_depth.reserve(old_levels + 1);
        degree_by_depth.push_back(old_degree_by_depth.empty() ? M : old_degree_by_depth.front());
        degree_by_depth.insert(degree_by_depth.end(),
                               old_degree_by_depth.begin(),
                               old_degree_by_depth.end());
        initialize_layout();

        for (PointId id = 0; id < num_points; ++id)
        {
            const std::uint32_t remapped = static_cast<std::uint32_t>(leaf_ids[id]) + old_leaf_offset;
            if (remapped >= tree.num_bins)
                throw Error("dynamic expansion produced an invalid leaf ID");
            leaf_ids[id] = static_cast<std::uint16_t>(remapped);
        }

        degree.assign(static_cast<std::size_t>(point_capacity) * tree.levels, 0);
        edge_records.assign(static_cast<std::size_t>(point_capacity) * capacity_per_point,
                            make_edge_record(kInvalidPointId, 0));
        auto copy_list = [&](PointId id, std::uint32_t old_depth, std::uint32_t new_depth) {
            const std::size_t old_degree_index = static_cast<std::size_t>(id) * old_levels + old_depth;
            const std::uint32_t copied = std::min<std::uint32_t>(
                std::min<std::uint32_t>(old_degree[old_degree_index], old_degree_by_depth[old_depth]),
                capacity_at_depth(new_depth));
            degree[degree_index(id, new_depth)] = static_cast<std::uint16_t>(copied);
            const std::size_t old_base = static_cast<std::size_t>(id) * old_capacity_per_point +
                                         old_depth_offsets[old_depth];
            for (std::uint32_t slot = 0; slot < copied; ++slot)
                edge_records[neighbor_index(id, new_depth, slot)] = old_edge_records[old_base + slot];
        };
        for (PointId id = 0; id < point_capacity; ++id)
        {
            copy_list(id, 0, 0);
            for (std::uint32_t old_depth = 0; old_depth < old_levels; ++old_depth)
                copy_list(id, old_depth, old_depth + 1);
        }

        tree.reconstruct_members(leaf_ids, internal_to_external, num_points);

        cut_token_by_leaf.assign(tree.num_bins, 0);
        cut_count_by_leaf.assign(tree.num_bins, 0);
        cut_depths_by_leaf.assign(static_cast<std::size_t>(tree.num_bins) * tree.levels, 0);
        cut_token = 1;
        reverse_adjacency.clear();
        reverse_adjacency_ready = false;
        build_edge_metadata();
        if (had_reverse)
            build_reverse_adjacency();

        ++stats.attr_expand_count;
        if (left_expansion)
            ++stats.attr_expand_left_count;
        else
            ++stats.attr_expand_right_count;
        stats.attr_expand_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
}
