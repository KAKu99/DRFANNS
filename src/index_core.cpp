#include "index_internal.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <thread>
#include <sys/mman.h>

namespace rfanns
{
    namespace
    {
        constexpr std::size_t kHugePageAlignment = 1U << 21;

        std::size_t round_up(std::size_t value, std::size_t alignment)
        {
            if (value > std::numeric_limits<std::size_t>::max() - alignment + 1)
                throw Error("aligned allocation size overflow");
            return ((value + alignment - 1) / alignment) * alignment;
        }

        bool contains(const std::vector<PointId> &values, PointId id)
        {
            return std::find(values.begin(), values.end(), id) != values.end();
        }
    }

    std::uint64_t splitmix64(std::uint64_t value) noexcept
    {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    AlignedFloatBuffer::~AlignedFloatBuffer()
    {
        std::free(data_);
    }

    void AlignedFloatBuffer::allocate(std::size_t floats)
    {
        std::free(data_);
        data_ = nullptr;
        allocated_floats_ = 0;
        if (floats == 0)
            return;
        const std::size_t bytes = round_up(floats * sizeof(float), kHugePageAlignment);
        data_ = static_cast<float *>(std::aligned_alloc(kHugePageAlignment, bytes));
        if (data_ == nullptr)
            throw Error("failed to allocate aligned vector storage");
        std::memset(data_, 0, bytes);
        (void)madvise(data_, bytes, MADV_HUGEPAGE);
        allocated_floats_ = bytes / sizeof(float);
    }

    void AlignedFloatBuffer::grow(std::size_t old_floats, std::size_t new_floats)
    {
        if (new_floats <= allocated_floats_)
            return;
        AlignedFloatBuffer replacement;
        replacement.allocate(new_floats);
        if (data_ != nullptr && old_floats != 0)
            std::copy_n(data_, std::min(old_floats, allocated_floats_), replacement.data_);
        std::swap(data_, replacement.data_);
        std::swap(allocated_floats_, replacement.allocated_floats_);
    }

    std::uint32_t ThreadScratch::next_tag()
    {
        if (++tag == 0)
        {
            std::fill(visited.begin(), visited.end(), 0);
            tag = 1;
        }
        return tag;
    }

    IndexImpl::IndexImpl(std::shared_ptr<const DenseDataset> data, std::uint64_t random_seed)
        : dataset(std::move(data)), seed(random_seed)
    {
        if (!dataset)
            throw Error("dataset is null");
    }

    bool IndexImpl::search_prepared() const noexcept
    {
        return !internal_vectors.empty() &&
               attribute_keys.size() == point_capacity &&
               edge_records.size() == static_cast<std::size_t>(point_capacity) * capacity_per_point &&
               edge_scope_low.size() == edge_list_capacity_count() &&
               edge_scope_high.size() == edge_list_capacity_count();
    }

    void IndexImpl::initialize_distance()
    {
        if (dimension == 0)
            throw Error("index dimension is zero");
        space = std::make_unique<hnswlib::L2Space>(dimension);
        distance_function = space->get_dist_func();
        distance_parameter = space->get_dist_func_param();
    }

    void IndexImpl::initialize_layout()
    {
        if (degree_by_depth.empty())
            degree_by_depth.assign(tree.levels, M);
        if (degree_by_depth.size() != tree.levels)
            throw Error("degree_by_depth size does not match tree levels");
        depth_offsets.assign(tree.levels, 0);
        std::size_t total = 0;
        for (std::uint32_t depth = 0; depth < tree.levels; ++depth)
        {
            if (degree_by_depth[depth] == 0 ||
                degree_by_depth[depth] > std::numeric_limits<std::uint16_t>::max())
                throw Error("degree capacity must be in [1, 65535]");
            depth_offsets[depth] = total;
            total += degree_by_depth[depth];
        }
        if (total > std::numeric_limits<std::uint32_t>::max())
            throw Error("per-point graph capacity is too large");
        capacity_per_point = static_cast<std::uint32_t>(total);
    }

    void IndexImpl::materialize_identity_id_map()
    {
        if (!internal_to_external.empty())
            return;
        internal_to_external.resize(point_capacity, kInvalidPointId);
        for (PointId id = 0; id < num_points; ++id)
            internal_to_external[id] = id;
        rebuild_external_map();
    }

    void IndexImpl::rebuild_external_map()
    {
        external_to_internal.clear();
        if (internal_to_external.empty())
            return;
        ExternalId maximum = 0;
        for (PointId id = 0; id < num_points; ++id)
        {
            const ExternalId external = internal_to_external[id];
            if (external != kInvalidPointId)
                maximum = std::max(maximum, external);
        }
        external_to_internal.assign(static_cast<std::size_t>(maximum) + 1, kInvalidPointId);
        for (PointId id = 0; id < num_points; ++id)
        {
            const ExternalId external = internal_to_external[id];
            if (external == kInvalidPointId)
                continue;
            if (external_to_internal[external] != kInvalidPointId)
                throw Error("ID map contains duplicate external IDs");
            external_to_internal[external] = id;
        }
    }

    bool IndexImpl::slot_is_bound(PointId id) const noexcept
    {
        return id < num_points &&
               (internal_to_external.empty() ||
                internal_to_external[id] != kInvalidPointId);
    }

    PointId IndexImpl::to_internal(ExternalId id) const
    {
        if (internal_to_external.empty())
            return id < num_points ? id : kInvalidPointId;
        if (id >= external_to_internal.size())
            return kInvalidPointId;
        return external_to_internal[id];
    }

    ExternalId IndexImpl::to_external(PointId id) const
    {
        if (id >= num_points)
            throw Error("internal ID is out of range");
        return internal_to_external.empty() ? id : internal_to_external[id];
    }

    void IndexImpl::reserve(std::uint32_t capacity)
    {
        if (capacity <= point_capacity)
            return;
        const std::uint32_t old_capacity = point_capacity;
        point_capacity = capacity;
        attributes.resize(point_capacity, 0);
        attribute_keys.resize(point_capacity, 0);
        leaf_ids.resize(point_capacity, 0);
        visited.resize(point_capacity, 0);
        degree.resize(static_cast<std::size_t>(point_capacity) * tree.levels, 0);
        edge_records.resize(static_cast<std::size_t>(point_capacity) * capacity_per_point,
                         make_edge_record(kInvalidPointId, 0));
        if (!internal_to_external.empty())
            internal_to_external.resize(point_capacity, kInvalidPointId);
        if (!internal_vectors.empty())
        {
            internal_vectors.grow(static_cast<std::size_t>(old_capacity) * vector_stride,
                                  static_cast<std::size_t>(point_capacity) * vector_stride);
        }
        if (!edge_scope_low.empty())
        {
            edge_scope_low.resize(edge_list_capacity_count(), 0);
            edge_scope_high.resize(edge_list_capacity_count(), 0);
        }
        if (reverse_adjacency_ready)
            reverse_adjacency.resize(edge_list_capacity_count());
    }

    void IndexImpl::build_internal_vectors()
    {
        vector_stride = ((static_cast<std::size_t>(dimension) + 7) / 8) * 8;
        internal_vectors.allocate(static_cast<std::size_t>(point_capacity) * vector_stride);
        for (PointId id = 0; id < num_points; ++id)
        {
            if (!slot_is_bound(id))
                continue;
            const ExternalId source = to_external(id);
            if (source >= dataset->size())
                throw Error("ID map references a vector outside the supplied dataset");
            float *destination = internal_vectors.data() + static_cast<std::size_t>(id) * vector_stride;
            std::copy_n(dataset->row(source), dimension, destination);
        }
    }

    void IndexImpl::build_edge_metadata()
    {
        attribute_keys.resize(point_capacity, 0);
        for (PointId id = 0; id < num_points; ++id)
            attribute_keys[id] = attribute_key(attributes[id]);
        if (edge_records.size() != static_cast<std::size_t>(point_capacity) * capacity_per_point)
            throw Error("edge-record array size does not match graph layout");
        edge_scope_low.assign(edge_list_capacity_count(), std::numeric_limits<AttrKey>::max());
        edge_scope_high.assign(edge_list_capacity_count(), 0);
        for (PointId id = 0; id < num_points; ++id)
        {
            for (std::uint32_t depth = 0; depth < tree.levels; ++depth)
                rebuild_edge_scope(id, depth);
        }
    }

    void IndexImpl::prepare_search()
    {
        if (search_prepared())
            return;
        if (point_capacity < num_points)
            point_capacity = num_points;
        initialize_distance();
        build_internal_vectors();
        build_edge_metadata();
        visited.assign(point_capacity, 0);
        cut_token_by_leaf.assign(tree.num_bins, 0);
        cut_count_by_leaf.assign(tree.num_bins, 0);
        cut_depths_by_leaf.assign(static_cast<std::size_t>(tree.num_bins) * tree.levels, 0);
    }

    void IndexImpl::prepare_updates()
    {
        prepare_search();
        if (!reverse_adjacency_ready)
            build_reverse_adjacency();
    }

    void IndexImpl::rebuild_edge_scope(PointId id, std::uint32_t depth)
    {
        if (edge_scope_low.empty())
            return;
        const std::size_t index = edge_list_index(id, depth);
        AttrKey low = std::numeric_limits<AttrKey>::max();
        AttrKey high = 0;
        const std::uint16_t count = get_degree(id, depth);
        for (std::uint16_t slot = 0; slot < count; ++slot)
        {
            const AttrKey attribute =
                edge_attribute(edge_records[neighbor_index(id, depth, slot)]);
            low = std::min(low, attribute);
            high = std::max(high, attribute);
        }
        edge_scope_low[index] = low;
        edge_scope_high[index] = high;
    }

    std::vector<PointId> IndexImpl::current_neighbors(PointId id, std::uint32_t depth) const
    {
        std::vector<PointId> result;
        const std::uint16_t count = get_degree(id, depth);
        result.reserve(count);
        for (std::uint16_t slot = 0; slot < count; ++slot)
            result.push_back(get_neighbor(id, depth, slot));
        return result;
    }

    std::vector<PointId> IndexImpl::select_hnsw_neighbors(
        PointId center,
        const std::vector<PointId> &candidates,
        std::uint32_t limit) const
    {
        std::vector<DistanceId> scored;
        scored.reserve(candidates.size());
        for (PointId candidate : candidates)
        {
            if (candidate == center || candidate == kInvalidPointId || candidate >= num_points)
                continue;
            if (std::find_if(scored.begin(), scored.end(), [&](const DistanceId &item) {
                    return item.second == candidate;
                }) == scored.end())
                scored.emplace_back(distance(center, candidate), candidate);
        }
        return select_hnsw_neighbors_scored(center, scored, limit);
    }

    std::vector<PointId> IndexImpl::select_hnsw_neighbors_scored(
        PointId center,
        const std::vector<DistanceId> &candidates,
        std::uint32_t limit) const
    {
        std::vector<DistanceId> ordered;
        ordered.reserve(candidates.size());
        for (const DistanceId &candidate : candidates)
        {
            if (candidate.second == center || candidate.second == kInvalidPointId ||
                candidate.second >= num_points)
                continue;
            if (std::find_if(ordered.begin(), ordered.end(), [&](const DistanceId &item) {
                    return item.second == candidate.second;
                }) == ordered.end())
                ordered.push_back(candidate);
        }
        return select_hnsw_neighbors_scored(center, ordered, limit);
    }

    std::vector<PointId> IndexImpl::select_hnsw_neighbors_scored(
        PointId,
        std::vector<DistanceId> &candidates,
        std::uint32_t limit) const
    {
        // Construction search already guarantees unique, valid, non-self
        // candidates.  Sort that scored frontier in place so the hot build
        // path performs neither distance recomputation nor a second candidate
        // copy.  The const overload above remains the defensive API used by
        // tests and callers without the construction invariant.
        std::sort(candidates.begin(), candidates.end());
        std::vector<PointId> result;
        if (candidates.size() < limit)
        {
            result.reserve(candidates.size());
            for (const DistanceId &candidate : candidates)
                result.push_back(candidate.second);
            return result;
        }

        std::vector<PointId> selected;
        selected.reserve(std::min<std::size_t>(limit, candidates.size()));
        for (const DistanceId &candidate : candidates)
        {
            if (selected.size() >= limit)
                break;
            bool accepted = true;
            for (PointId chosen : selected)
            {
                if (distance(candidate.second, chosen) < candidate.first)
                {
                    accepted = false;
                    break;
                }
            }
            if (accepted)
                selected.push_back(candidate.second);
        }
        return selected;
    }

    NeighborRewriteSummary IndexImpl::write_neighbor_list(
        PointId owner,
        std::uint32_t depth,
        const std::vector<PointId> &list)
    {
        if (list.size() > capacity_at_depth(depth))
            throw Error("neighbor list exceeds the depth capacity");
        const std::uint16_t old_count = get_degree(owner, depth);
        const std::size_t base = neighbor_index(owner, depth, 0);
        NeighborRewriteSummary summary;
        if (reverse_adjacency_ready)
        {
            for (std::uint16_t slot = 0; slot < old_count; ++slot)
            {
                const PointId target = edge_destination(edge_records[base + slot]);
                if (contains(list, target))
                    continue;
                auto &owners = reverse_adjacency[edge_list_index(target, depth)];
                const auto it = std::find(owners.begin(), owners.end(), owner);
                if (it != owners.end())
                {
                    owners.erase(it);
                    --summary.reverse_edge_count_change;
                    ++summary.reverse_updates;
                }
            }
            for (PointId target : list)
            {
                bool existed = false;
                for (std::uint16_t slot = 0; slot < old_count; ++slot)
                {
                    if (edge_destination(edge_records[base + slot]) == target)
                    {
                        existed = true;
                        break;
                    }
                }
                if (existed)
                    continue;
                auto &owners = reverse_adjacency[edge_list_index(target, depth)];
                if (!contains(owners, owner))
                {
                    owners.push_back(owner);
                    ++summary.reverse_edge_count_change;
                    ++summary.reverse_updates;
                }
            }
        }

        AttrKey scope_low = std::numeric_limits<AttrKey>::max();
        AttrKey scope_high = 0;
        degree[degree_index(owner, depth)] = static_cast<std::uint16_t>(list.size());
        for (std::uint32_t slot = 0; slot < capacity_at_depth(depth); ++slot)
        {
            const PointId target = slot < list.size() ? list[slot] : kInvalidPointId;
            edge_records[base + slot] = target == kInvalidPointId
                                          ? make_edge_record(target, 0)
                                          : make_edge_record(target, attribute_keys[target]);
            if (slot < list.size() && !edge_scope_low.empty())
            {
                scope_low = std::min(scope_low, attribute_keys[target]);
                scope_high = std::max(scope_high, attribute_keys[target]);
            }
        }
        if (!edge_scope_low.empty())
        {
            const std::size_t index = edge_list_index(owner, depth);
            edge_scope_low[index] = scope_low;
            edge_scope_high[index] = scope_high;
        }
        summary.forward_lists_rewritten = 1U;
        return summary;
    }

    void IndexImpl::commit_neighbor_rewrite_stats(const NeighborRewriteSummary &summary,
                                                  double batch_seconds)
    {
        if (summary.reverse_edge_count_change < 0)
        {
            const std::uint64_t removed =
                static_cast<std::uint64_t>(-summary.reverse_edge_count_change);
            if (removed > stats.reverse_adjacency_edges)
                throw Error("reverse adjacency edge accounting underflow");
            stats.reverse_adjacency_edges -= removed;
        }
        else
        {
            stats.reverse_adjacency_edges +=
                static_cast<std::uint64_t>(summary.reverse_edge_count_change);
        }
        stats.reverse_adjacency_updates += summary.reverse_updates;
        stats.forward_lists_rewritten += summary.forward_lists_rewritten;
        if (summary.reverse_updates != 0)
            stats.reverse_adjacency_update_seconds += batch_seconds;
        if (summary.forward_lists_rewritten != 0)
            stats.forward_list_rewrite_seconds += batch_seconds;
    }

    void IndexImpl::build_reverse_adjacency(std::uint32_t threads)
    {
        const auto started = std::chrono::steady_clock::now();
        reverse_adjacency.assign(edge_list_capacity_count(), {});
        std::vector<std::uint32_t> counts(edge_list_capacity_count(), 0);
        std::vector<std::uint64_t> totals(tree.levels, 0);
        if (threads == 0)
            threads = std::max(1U, std::thread::hardware_concurrency());
        const std::uint32_t worker_count = std::max(
            1U, std::min<std::uint32_t>(threads, tree.levels));
        std::atomic<std::uint32_t> next_depth{0};
        std::atomic<bool> stop{false};
        std::exception_ptr worker_error;
        std::mutex error_mutex;
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::uint32_t worker = 0; worker < worker_count; ++worker)
        {
            workers.emplace_back([&, this]() {
                try
                {
                    while (!stop.load(std::memory_order_relaxed))
                    {
                        const std::uint32_t depth =
                            next_depth.fetch_add(1, std::memory_order_relaxed);
                        if (depth >= tree.levels)
                            break;
                        std::uint64_t total = 0;
                        for (PointId owner = 0; owner < num_points; ++owner)
                        {
                            for (std::uint16_t slot = 0;
                                 slot < get_degree(owner, depth);
                                 ++slot)
                            {
                                const PointId target = get_neighbor(owner, depth, slot);
                                ++counts[edge_list_index(target, depth)];
                                ++total;
                            }
                        }
                        for (PointId target = 0; target < num_points; ++target)
                        {
                            const std::size_t index = edge_list_index(target, depth);
                            reverse_adjacency[index].reserve(counts[index]);
                        }
                        for (PointId owner = 0; owner < num_points; ++owner)
                        {
                            for (std::uint16_t slot = 0;
                                 slot < get_degree(owner, depth);
                                 ++slot)
                            {
                                const PointId target = get_neighbor(owner, depth, slot);
                                reverse_adjacency[edge_list_index(target, depth)].push_back(owner);
                            }
                        }
                        totals[depth] = total;
                    }
                }
                catch (...)
                {
                    {
                        std::lock_guard<std::mutex> guard(error_mutex);
                        if (!worker_error)
                            worker_error = std::current_exception();
                    }
                    stop.store(true, std::memory_order_relaxed);
                }
            });
        }
        for (auto &worker : workers)
            worker.join();
        if (worker_error)
            std::rethrow_exception(worker_error);

        const std::uint64_t total =
            std::accumulate(totals.begin(), totals.end(), std::uint64_t{0});
        reverse_adjacency_ready = true;
        stats.reverse_adjacency_edges = total;
        stats.reverse_adjacency_updates = 0;
        stats.reverse_adjacency_update_seconds = 0.0;
        stats.reverse_adjacency_build_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    std::uint32_t IndexImpl::next_visited_tag()
    {
        if (++visited_tag == 0)
        {
            std::fill(visited.begin(), visited.end(), 0);
            visited_tag = 1;
        }
        return visited_tag;
    }

    void IndexImpl::begin_cut_cache_query()
    {
        if (++cut_token == 0)
        {
            std::fill(cut_token_by_leaf.begin(), cut_token_by_leaf.end(), 0);
            cut_token = 1;
        }
    }

    const std::uint8_t *IndexImpl::cut_depths(std::uint16_t leaf,
                                              std::uint16_t query_l,
                                              std::uint16_t query_r,
                                              std::uint8_t &count)
    {
        if (leaf >= tree.num_bins)
            throw Error("leaf id is outside the tree while selecting edges");
        const std::size_t base = static_cast<std::size_t>(leaf) * tree.levels;
        if (cut_token_by_leaf[leaf] == cut_token)
        {
            count = cut_count_by_leaf[leaf];
            return cut_depths_by_leaf.data() + base;
        }

        count = 0;
        const auto &path = tree.path_by_leaf[leaf];
        std::uint32_t start = 0;
        while (start < tree.levels)
        {
            std::uint32_t depth = start;
            while (depth + 1 < tree.levels &&
                   tree.overlap(path[depth], query_l, query_r) ==
                       tree.overlap(path[depth + 1], query_l, query_r))
                ++depth;
            cut_depths_by_leaf[base + count++] = static_cast<std::uint8_t>(depth);
            if (tree.node_inside_query(path[depth], query_l, query_r) || depth + 1 >= tree.levels)
                break;
            start = depth + 1;
        }
        cut_token_by_leaf[leaf] = cut_token;
        cut_count_by_leaf[leaf] = count;
        return cut_depths_by_leaf.data() + base;
    }

    void IndexImpl::validate() const
    {
        if (num_points == 0 || point_capacity < num_points)
            throw Error("invalid point count or capacity");
        if (tree.num_bins == 0 || tree.levels == 0)
            throw Error("invalid range tree metadata");
        if (attributes.size() < point_capacity || attribute_keys.size() < point_capacity ||
            leaf_ids.size() < point_capacity)
            throw Error("point metadata is smaller than point capacity");
        if (!internal_to_external.empty() && internal_to_external.size() < point_capacity)
            throw Error("ID map is smaller than point capacity");
        if (degree.size() != static_cast<std::size_t>(point_capacity) * tree.levels)
            throw Error("degree array size does not match graph layout");
        if (edge_records.size() != static_cast<std::size_t>(point_capacity) * capacity_per_point)
            throw Error("edge-record array size does not match graph layout");
        if (edge_scope_low.empty() != edge_scope_high.empty())
            throw Error("edge-scope arrays are only partially initialized");
        if (!edge_scope_low.empty() &&
            (edge_scope_low.size() != edge_list_capacity_count() ||
             edge_scope_high.size() != edge_list_capacity_count()))
            throw Error("edge-scope array size does not match graph layout");

        std::vector<std::uint32_t> memberships(num_points, 0);
        for (const auto &node : tree.nodes)
        {
            for (PointId id : node.members)
            {
                if (!slot_is_bound(id))
                    throw Error("tree contains an invalid or inactive member");
                if (leaf_ids[id] >= tree.num_bins || node.depth < 0 ||
                    static_cast<std::uint32_t>(node.depth) >= tree.levels ||
                    tree.path_by_leaf[leaf_ids[id]][node.depth] != node.node_id)
                    throw Error("tree member is stored outside its leaf-to-root path");
                ++memberships[id];
            }
        }
        std::uint32_t live_count = 0;
        for (PointId id = 0; id < num_points; ++id)
        {
            const bool live = slot_is_bound(id);
            live_count += live ? 1U : 0U;
            if (live && memberships[id] != tree.levels)
                throw Error("an active point does not belong to one segment per level");
            if (!live && memberships[id] != 0)
                throw Error("an inactive point remains in tree members");
            if (!live)
            {
                for (std::uint32_t depth = 0; depth < tree.levels; ++depth)
                {
                    if (get_degree(id, depth) != 0)
                        throw Error("an inactive point retains outgoing edges");
                    for (std::uint32_t slot = 0; slot < capacity_at_depth(depth); ++slot)
                    {
                        const EdgeRecord record = edge_records[neighbor_index(id, depth, slot)];
                        if (edge_destination(record) != kInvalidPointId || edge_attribute(record) != 0)
                            throw Error("an inactive point retains an edge-record slot");
                    }
                }
                continue;
            }
            for (std::uint32_t depth = 0; depth < tree.levels; ++depth)
            {
                const std::uint16_t count = get_degree(id, depth);
                if (count > capacity_at_depth(depth))
                    throw Error("degree exceeds depth capacity");
                std::unordered_set<PointId> unique;
                AttrKey expected_scope_low = std::numeric_limits<AttrKey>::max();
                AttrKey expected_scope_high = 0;
                for (std::uint16_t slot = 0; slot < count; ++slot)
                {
                    const EdgeRecord record = edge_records[neighbor_index(id, depth, slot)];
                    const PointId neighbor = edge_destination(record);
                    if (neighbor == id || !slot_is_bound(neighbor))
                        throw Error("neighbor list contains an invalid point");
                    if (edge_attribute(record) != attribute_keys[neighbor])
                        throw Error("edge-record attribute is inconsistent with its destination");
                    if (!unique.insert(neighbor).second)
                        throw Error("neighbor list contains a duplicate");
                    const auto a = tree.path_by_leaf[leaf_ids[id]][depth];
                    const auto b = tree.path_by_leaf[leaf_ids[neighbor]][depth];
                    if (a != b)
                        throw Error("edge crosses segment boundaries at its depth");
                    expected_scope_low = std::min(expected_scope_low, edge_attribute(record));
                    expected_scope_high = std::max(expected_scope_high, edge_attribute(record));
                }
                for (std::uint32_t slot = count; slot < capacity_at_depth(depth); ++slot)
                {
                    const EdgeRecord record = edge_records[neighbor_index(id, depth, slot)];
                    if (edge_destination(record) != kInvalidPointId || edge_attribute(record) != 0)
                        throw Error("unused edge-record slot is not invalid");
                }
                if (!edge_scope_low.empty())
                {
                    const std::size_t list_index = edge_list_index(id, depth);
                    if (edge_scope_low[list_index] != expected_scope_low ||
                        edge_scope_high[list_index] != expected_scope_high)
                        throw Error("edge scope is inconsistent with its adjacency list");
                }
            }
        }
        if (live_count != active_points)
            throw Error("active point count does not match ID bindings");
        if (!internal_to_external.empty())
        {
            std::unordered_set<ExternalId> unique;
            for (PointId id = 0; id < num_points; ++id)
            {
                const ExternalId external = internal_to_external[id];
                if (external == kInvalidPointId)
                    continue;
                if (!unique.insert(external).second)
                    throw Error("ID map contains a duplicate");
                if (external >= external_to_internal.size() ||
                    external_to_internal[external] != id)
                    throw Error("internal and external ID maps are inconsistent");
            }
            for (ExternalId external = 0; external < external_to_internal.size(); ++external)
            {
                const PointId id = external_to_internal[external];
                if (id == kInvalidPointId)
                    continue;
                if (id >= num_points || internal_to_external[id] != external)
                    throw Error("external ID map contains a stale binding");
            }
        }
        if (reverse_adjacency_ready)
        {
            if (reverse_adjacency.size() != edge_list_capacity_count())
                throw Error("reverse adjacency size does not match graph layout");
            for (PointId owner = 0; owner < num_points; ++owner)
            {
                if (!slot_is_bound(owner))
                    continue;
                for (std::uint32_t depth = 0; depth < tree.levels; ++depth)
                {
                    for (std::uint16_t slot = 0; slot < get_degree(owner, depth); ++slot)
                    {
                        const PointId target = get_neighbor(owner, depth, slot);
                        const auto &owners = reverse_adjacency[edge_list_index(target, depth)];
                        if (!contains(owners, owner))
                            throw Error("reverse adjacency is missing a graph edge");
                    }
                }
            }
            for (PointId target = 0; target < point_capacity; ++target)
            {
                for (std::uint32_t depth = 0; depth < tree.levels; ++depth)
                {
                    const auto &owners = reverse_adjacency[edge_list_index(target, depth)];
                    std::unordered_set<PointId> unique_owners;
                    for (PointId owner : owners)
                    {
                        if (!slot_is_bound(target) || !slot_is_bound(owner))
                            throw Error("reverse adjacency contains an inactive or invalid edge");
                        if (!unique_owners.insert(owner).second)
                            throw Error("reverse adjacency contains a duplicate owner");
                        bool found = false;
                        for (std::uint16_t slot = 0; slot < get_degree(owner, depth); ++slot)
                        {
                            if (get_neighbor(owner, depth, slot) == target)
                            {
                                found = true;
                                break;
                            }
                        }
                        if (!found)
                            throw Error("reverse adjacency contains a stale owner");
                    }
                }
            }
        }
    }

    Index::Index(std::unique_ptr<IndexImpl> impl) : impl_(std::move(impl)) {}
    Index::Index(Index &&) noexcept = default;
    Index &Index::operator=(Index &&) noexcept = default;
    Index::~Index() = default;

    void Index::prepare_search() { impl_->prepare_search(); }
    void Index::prepare_updates() { impl_->prepare_updates(); }
    void Index::reserve(std::uint32_t capacity) { impl_->reserve(capacity); }

    std::vector<SearchResult> Index::search(const float *query,
                                            std::uint32_t query_id,
                                            AttributeRange range,
                                            const SearchOptions &options)
    {
        return impl_->search(query, query_id, range, options);
    }

    PointId Index::insert(ExternalId external_id,
                          const float *vector,
                          Attribute attribute,
                          const InsertOptions &options)
    {
        return impl_->insert(external_id, vector, attribute, options);
    }

    bool Index::erase(ExternalId external_id)
    {
        return impl_->erase(external_id);
    }

    void Index::save(const std::string &index_path, const std::string &id_map_path) const
    {
        impl_->save(index_path);
        if (!id_map_path.empty())
            save_id_map(id_map_path, current_id_map());
    }

    void Index::validate() const { impl_->validate(); }

    IndexMetadata Index::metadata() const
    {
        return {impl_->num_points,
                impl_->active_points,
                impl_->dimension,
                impl_->M,
                impl_->ef_construction,
                impl_->tree.num_bins,
                impl_->tree.levels,
                impl_->tree.attr_min,
                impl_->tree.attr_max};
    }

    const UpdateStats &Index::update_stats() const { return impl_->stats; }

    std::vector<ExternalId> Index::current_id_map() const
    {
        std::vector<ExternalId> result(impl_->num_points, kInvalidPointId);
        if (impl_->internal_to_external.empty())
        {
            std::iota(result.begin(), result.end(), 0U);
            return result;
        }
        std::copy_n(impl_->internal_to_external.begin(), impl_->num_points, result.begin());
        return result;
    }
}
