#include "index_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

namespace rfanns
{
    namespace
    {
        using DistanceId = IndexImpl::DistanceId;

        template <typename Compare>
        void heap_push(std::vector<DistanceId> &heap,
                       const DistanceId &value,
                       Compare compare)
        {
            heap.push_back(value);
            std::push_heap(heap.begin(), heap.end(), compare);
        }

        template <typename Compare>
        DistanceId heap_pop(std::vector<DistanceId> &heap, Compare compare)
        {
            const DistanceId result = heap.front();
            std::pop_heap(heap.begin(), heap.end(), compare);
            heap.pop_back();
            return result;
        }

        void set_builder_neighbors(IndexImpl &index,
                                   PointId id,
                                   std::uint32_t depth,
                                   const std::vector<PointId> &neighbors)
        {
            if (neighbors.size() > index.capacity_at_depth(depth))
                throw Error("builder produced an oversized neighbor list");
            for (std::uint32_t slot = 0; slot < index.capacity_at_depth(depth); ++slot)
            {
                const PointId target =
                    slot < neighbors.size() ? neighbors[slot] : kInvalidPointId;
                index.edge_records[index.neighbor_index(id, depth, slot)] =
                    target == kInvalidPointId
                        ? make_edge_record(target, 0)
                        : make_edge_record(target, index.attribute_keys[target]);
            }
            index.degree[index.degree_index(id, depth)] =
                static_cast<std::uint16_t>(neighbors.size());
        }

        class SequentialGraphBuilder
        {
        public:
            explicit SequentialGraphBuilder(IndexImpl &index)
                : index_(index),
                  bootstrap_seeds_(index.tree.nodes.size(), kInvalidPointId)
            {
            }

            void build()
            {
                ThreadScratch scratch(index_.num_points);
                for (PointId id = 0; id < index_.num_points; ++id)
                    insert_path(id, scratch);
            }

        private:
            IndexImpl &index_;
            std::vector<PointId> bootstrap_seeds_;

            void insert_path(PointId id, ThreadScratch &scratch)
            {
                const auto &path = index_.tree.path_by_leaf[index_.leaf_ids[id]];
                PointId routed_seed = kInvalidPointId;
                std::int32_t inherited_node_id = -1;
                std::vector<IndexImpl::DistanceId> inherited_candidates;
                for (std::size_t position = path.size(); position-- > 0;)
                {
                    const std::int32_t node_id = path[position];
                    auto &node = index_.tree.nodes[node_id];
                    const std::uint32_t depth = static_cast<std::uint32_t>(node.depth);
                    if (bootstrap_seeds_[node_id] == kInvalidPointId)
                    {
                        set_builder_neighbors(index_, id, depth, {});
                        bootstrap_seeds_[node_id] = id;
                        routed_seed = kInvalidPointId;
                        inherited_node_id = -1;
                        inherited_candidates.clear();
                        continue;
                    }

                    const PointId seed = routed_seed != kInvalidPointId
                                             ? routed_seed
                                             : bootstrap_seeds_[node_id];
                    search(node_id,
                           id,
                           seed,
                           inherited_node_id,
                           inherited_candidates,
                           scratch,
                           scratch.candidate_distances);
                    if (scratch.candidate_distances.empty())
                        throw Error("nonempty segment search returned no construction candidate");
                    scratch.selected = index_.select_hnsw_neighbors_scored(
                        id, scratch.candidate_distances, index_.capacity_at_depth(depth));
                    // The scored selector orders by (distance, id), which is
                    // exactly the former full-scan routing tie rule.
                    routed_seed = scratch.candidate_distances.front().second;
                    set_builder_neighbors(index_, id, depth, scratch.selected);
                    for (PointId neighbor : scratch.selected)
                    {
                        std::vector<PointId> reciprocal = index_.current_neighbors(neighbor, depth);
                        if (std::find(reciprocal.begin(), reciprocal.end(), id) == reciprocal.end())
                            reciprocal.push_back(id);
                        if (reciprocal.size() > index_.capacity_at_depth(depth))
                        {
                            reciprocal = index_.select_hnsw_neighbors(
                                neighbor, reciprocal, index_.capacity_at_depth(depth));
                        }
                        set_builder_neighbors(index_, neighbor, depth, reciprocal);
                    }
                    inherited_candidates.swap(scratch.candidate_distances);
                    inherited_node_id = node_id;
                }
            }

            void search(std::int32_t node_id,
                        PointId query,
                        PointId seed,
                        std::int32_t inherited_node_id,
                        const std::vector<IndexImpl::DistanceId> &inherited,
                        ThreadScratch &scratch,
                        std::vector<IndexImpl::DistanceId> &result_distances)
            {
                const auto &node = index_.tree.nodes[node_id];
                const std::uint32_t depth = static_cast<std::uint32_t>(node.depth);
                const std::uint32_t search_ef =
                    std::max(index_.ef_construction, index_.capacity_at_depth(depth));
                const std::uint32_t tag = scratch.next_tag();
                const bool skip_inherited_region = inherited_node_id >= 0 && !inherited.empty();
                const std::int32_t inherited_region_leaf_l = skip_inherited_region
                                                          ? index_.tree.nodes[inherited_node_id].leaf_l
                                                          : 0;
                const std::int32_t inherited_region_leaf_r = skip_inherited_region
                                                          ? index_.tree.nodes[inherited_node_id].leaf_r
                                                          : -1;

                using Pair = DistanceId;
                auto &candidates = scratch.candidate_heap;
                auto &top = scratch.top_heap;
                candidates.clear();
                top.clear();
                candidates.reserve(search_ef + 1);
                top.reserve(search_ef + 1);
                bool seed_inherited = false;
                float seed_distance = 0.0f;
                for (const Pair &candidate : inherited)
                {
                    if (scratch.visited[candidate.second] == tag)
                        continue;
                    scratch.visited[candidate.second] = tag;
                    top.push_back(candidate);
                    candidates.push_back(candidate);
                    if (candidate.second == seed)
                    {
                        seed_inherited = true;
                        seed_distance = candidate.first;
                    }
                }
                std::make_heap(top.begin(), top.end());
                std::make_heap(candidates.begin(), candidates.end(), std::greater<Pair>{});
                while (top.size() > search_ef)
                    (void)heap_pop(top, std::less<Pair>{});
                if (!seed_inherited)
                {
                    seed_distance = index_.distance(query, seed);
                    scratch.visited[seed] = tag;
                    heap_push(top, Pair{seed_distance, seed}, std::less<Pair>{});
                    if (top.size() > search_ef)
                        (void)heap_pop(top, std::less<Pair>{});
                }
                if (!seed_inherited)
                    heap_push(candidates,
                              Pair{seed_distance, seed},
                              std::greater<Pair>{});
                float lower_bound = top.front().first;

                while (!candidates.empty())
                {
                    const Pair current = heap_pop(candidates, std::greater<Pair>{});
                    if (top.size() >= search_ef && current.first > lower_bound)
                        break;
                    const std::uint16_t count = index_.get_degree(current.second, depth);
                    for (std::uint16_t slot = 0; slot < count; ++slot)
                    {
                        const PointId neighbor = index_.get_neighbor(current.second, depth, slot);
                        if (scratch.visited[neighbor] == tag)
                            continue;
                        scratch.visited[neighbor] = tag;
                        const std::int32_t neighbor_leaf = index_.leaf_ids[neighbor];
                        if (skip_inherited_region && neighbor_leaf >= inherited_region_leaf_l &&
                            neighbor_leaf <= inherited_region_leaf_r)
                            continue;
                        const float distance = index_.distance(query, neighbor);
                        if (top.size() < search_ef || distance < lower_bound)
                        {
                            heap_push(candidates,
                                      Pair{distance, neighbor},
                                      std::greater<Pair>{});
                            heap_push(top, Pair{distance, neighbor}, std::less<Pair>{});
                            if (top.size() > search_ef)
                                (void)heap_pop(top, std::less<Pair>{});
                            lower_bound = top.front().first;
                        }
                    }
                }

                result_distances.clear();
                result_distances.reserve(top.size());
                while (!top.empty())
                    result_distances.push_back(heap_pop(top, std::less<Pair>{}));
            }
        };

        class ParallelGraphBuilder
        {
        public:
            explicit ParallelGraphBuilder(IndexImpl &index)
                : index_(index),
                  point_locks_(std::make_unique<std::mutex[]>(index.num_points)),
                  completed_min_depth_(
                      std::make_unique<std::atomic<std::uint8_t>[]>(index.num_points)),
                  bootstrap_seeds_(
                      std::make_unique<std::atomic<std::int32_t>[]>(index.tree.nodes.size()))
            {
                const auto unpublished = static_cast<std::uint8_t>(index_.tree.levels);
                for (PointId id = 0; id < index_.num_points; ++id)
                    completed_min_depth_[id].store(unpublished, std::memory_order_relaxed);
                for (std::size_t i = 0; i < index_.tree.nodes.size(); ++i)
                    bootstrap_seeds_[i].store(kEmptySeed, std::memory_order_relaxed);
            }

            void build(std::uint32_t threads)
            {
                std::atomic<PointId> next{0};
                std::atomic<bool> stop{false};
                std::exception_ptr worker_error;
                std::mutex error_mutex;
                const std::uint32_t worker_count = std::min(threads, index_.num_points);
                std::vector<std::thread> workers;
                workers.reserve(worker_count);
                for (std::uint32_t worker = 0; worker < worker_count; ++worker)
                {
                    workers.emplace_back([&, this]() {
                        ThreadScratch scratch(index_.num_points);
                        try
                        {
                            while (!stop.load(std::memory_order_relaxed))
                            {
                                const PointId id = next.fetch_add(1, std::memory_order_relaxed);
                                if (id >= index_.num_points)
                                    break;
                                insert_path(id, scratch);
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
            }

        private:
            static constexpr std::int32_t kEmptySeed = -1;
            static constexpr std::int32_t kInitializingSeed = -2;

            IndexImpl &index_;
            std::unique_ptr<std::mutex[]> point_locks_;
            std::unique_ptr<std::atomic<std::uint8_t>[]> completed_min_depth_;
            // Parallel construction needs a transient first-published point per
            // segment. This state is discarded after build and is not persisted.
            std::unique_ptr<std::atomic<std::int32_t>[]> bootstrap_seeds_;

            bool is_published(PointId id, std::uint32_t depth) const
            {
                return id < index_.num_points &&
                       completed_min_depth_[id].load(std::memory_order_acquire) <= depth;
            }

            bool valid_routed_seed(PointId seed, std::uint32_t depth) const
            {
                return seed != kInvalidPointId && is_published(seed, depth);
            }

            std::int32_t wait_for_bootstrap(std::int32_t node_id) const
            {
                std::int32_t seed =
                    bootstrap_seeds_[node_id].load(std::memory_order_acquire);
                while (seed == kInitializingSeed)
                {
                    std::this_thread::yield();
                    seed = bootstrap_seeds_[node_id].load(std::memory_order_acquire);
                }
                return seed;
            }

            bool initialize_if_empty(std::int32_t node_id, PointId id, std::uint32_t depth)
            {
                const std::int32_t observed =
                    bootstrap_seeds_[node_id].load(std::memory_order_acquire);
                if (observed != kEmptySeed)
                {
                    if (observed == kInitializingSeed)
                        (void)wait_for_bootstrap(node_id);
                    return false;
                }
                std::int32_t expected = kEmptySeed;
                if (bootstrap_seeds_[node_id].compare_exchange_strong(
                        expected,
                        kInitializingSeed,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire))
                {
                    set_builder_neighbors(index_, id, depth, {});
                    completed_min_depth_[id].store(static_cast<std::uint8_t>(depth),
                                                   std::memory_order_release);
                    bootstrap_seeds_[node_id].store(static_cast<std::int32_t>(id),
                                                      std::memory_order_release);
                    return true;
                }
                if (expected == kInitializingSeed)
                    (void)wait_for_bootstrap(node_id);
                return false;
            }

            PointId fallback_seed(std::int32_t node_id, std::uint32_t depth) const
            {
                const std::int32_t stored = wait_for_bootstrap(node_id);
                if (stored >= 0)
                {
                    const PointId seed = static_cast<PointId>(stored);
                    if (valid_routed_seed(seed, depth))
                        return seed;
                }
                for (PointId member : index_.tree.nodes[node_id].members)
                {
                    if (valid_routed_seed(member, depth))
                        return member;
                }
                throw Error("nonempty parallel segment has no published seed");
            }

            void insert_path(PointId id, ThreadScratch &scratch)
            {
                const auto &path = index_.tree.path_by_leaf[index_.leaf_ids[id]];
                PointId routed_seed = kInvalidPointId;
                std::int32_t inherited_node_id = -1;
                std::vector<IndexImpl::DistanceId> inherited_candidates;
                for (std::size_t position = path.size(); position-- > 0;)
                {
                    const std::int32_t node_id = path[position];
                    const std::uint32_t depth =
                        static_cast<std::uint32_t>(index_.tree.nodes[node_id].depth);
                    if (initialize_if_empty(node_id, id, depth))
                    {
                        routed_seed = kInvalidPointId;
                        inherited_node_id = -1;
                        inherited_candidates.clear();
                        continue;
                    }

                    const PointId seed = valid_routed_seed(routed_seed, depth)
                                             ? routed_seed
                                             : fallback_seed(node_id, depth);
                    search(node_id,
                           id,
                           seed,
                           inherited_node_id,
                           inherited_candidates,
                           scratch,
                           scratch.candidate_distances);
                    if (scratch.candidate_distances.empty())
                        throw Error("nonempty parallel segment search returned no candidate");
                    scratch.selected = index_.select_hnsw_neighbors_scored(
                        id, scratch.candidate_distances, index_.capacity_at_depth(depth));
                    routed_seed = scratch.candidate_distances.front().second;
                    set_builder_neighbors(index_, id, depth, scratch.selected);
                    completed_min_depth_[id].store(static_cast<std::uint8_t>(depth),
                                                   std::memory_order_release);

                    for (PointId neighbor : scratch.selected)
                    {
                        std::lock_guard<std::mutex> guard(point_locks_[neighbor]);
                        std::vector<PointId> reciprocal = index_.current_neighbors(neighbor, depth);
                        if (std::find(reciprocal.begin(), reciprocal.end(), id) == reciprocal.end())
                            reciprocal.push_back(id);
                        if (reciprocal.size() > index_.capacity_at_depth(depth))
                        {
                            reciprocal = index_.select_hnsw_neighbors(
                                neighbor, reciprocal, index_.capacity_at_depth(depth));
                        }
                        set_builder_neighbors(index_, neighbor, depth, reciprocal);
                    }
                    inherited_candidates.swap(scratch.candidate_distances);
                    inherited_node_id = node_id;
                }
            }

            void search(std::int32_t node_id,
                        PointId query,
                        PointId seed,
                        std::int32_t inherited_node_id,
                        const std::vector<IndexImpl::DistanceId> &inherited,
                        ThreadScratch &scratch,
                        std::vector<IndexImpl::DistanceId> &result_distances)
            {
                const auto &node = index_.tree.nodes[node_id];
                const std::uint32_t depth = static_cast<std::uint32_t>(node.depth);
                const std::uint32_t search_ef =
                    std::max(index_.ef_construction, index_.capacity_at_depth(depth));
                const std::uint32_t tag = scratch.next_tag();
                const bool skip_inherited_region = inherited_node_id >= 0 && !inherited.empty();
                const std::int32_t inherited_region_leaf_l = skip_inherited_region
                                                          ? index_.tree.nodes[inherited_node_id].leaf_l
                                                          : 0;
                const std::int32_t inherited_region_leaf_r = skip_inherited_region
                                                          ? index_.tree.nodes[inherited_node_id].leaf_r
                                                          : -1;

                using Pair = DistanceId;
                auto &candidates = scratch.candidate_heap;
                auto &top = scratch.top_heap;
                candidates.clear();
                top.clear();
                candidates.reserve(search_ef + 1);
                top.reserve(search_ef + 1);
                bool seed_inherited = false;
                float seed_distance = 0.0f;
                for (const Pair &candidate : inherited)
                {
                    const PointId inherited_id = candidate.second;
                    if (!is_published(inherited_id, depth) ||
                        scratch.visited[inherited_id] == tag)
                        continue;
                    scratch.visited[inherited_id] = tag;
                    top.push_back(candidate);
                    candidates.push_back(candidate);
                    if (inherited_id == seed)
                    {
                        seed_inherited = true;
                        seed_distance = candidate.first;
                    }
                }
                std::make_heap(top.begin(), top.end());
                std::make_heap(candidates.begin(), candidates.end(), std::greater<Pair>{});
                while (top.size() > search_ef)
                    (void)heap_pop(top, std::less<Pair>{});
                if (!seed_inherited)
                {
                    seed_distance = index_.distance(query, seed);
                    scratch.visited[seed] = tag;
                    heap_push(top, Pair{seed_distance, seed}, std::less<Pair>{});
                    if (top.size() > search_ef)
                        (void)heap_pop(top, std::less<Pair>{});
                }
                if (!seed_inherited)
                    heap_push(candidates,
                              Pair{seed_distance, seed},
                              std::greater<Pair>{});
                float lower_bound = top.front().first;
                auto &snapshot = scratch.adjacency_snapshot;
                snapshot.clear();
                snapshot.reserve(index_.capacity_at_depth(depth));

                while (!candidates.empty())
                {
                    const Pair current = heap_pop(candidates, std::greater<Pair>{});
                    if (top.size() >= search_ef && current.first > lower_bound)
                        break;

                    snapshot.clear();
                    {
                        std::lock_guard<std::mutex> guard(point_locks_[current.second]);
                        const std::uint16_t count = index_.get_degree(current.second, depth);
                        for (std::uint16_t slot = 0; slot < count; ++slot)
                            snapshot.push_back(index_.get_neighbor(current.second, depth, slot));
                    }
                    for (PointId neighbor : snapshot)
                    {
                        if (scratch.visited[neighbor] == tag)
                            continue;
                        scratch.visited[neighbor] = tag;
                        const std::int32_t neighbor_leaf = index_.leaf_ids[neighbor];
                        if (skip_inherited_region && neighbor_leaf >= inherited_region_leaf_l &&
                            neighbor_leaf <= inherited_region_leaf_r)
                            continue;
                        const float distance = index_.distance(query, neighbor);
                        if (top.size() < search_ef || distance < lower_bound)
                        {
                            heap_push(candidates,
                                      Pair{distance, neighbor},
                                      std::greater<Pair>{});
                            heap_push(top, Pair{distance, neighbor}, std::less<Pair>{});
                            if (top.size() > search_ef)
                                (void)heap_pop(top, std::less<Pair>{});
                            lower_bound = top.front().first;
                        }
                    }
                }

                result_distances.clear();
                result_distances.reserve(top.size());
                while (!top.empty())
                    result_distances.push_back(heap_pop(top, std::less<Pair>{}));
            }

        };

        DomainOptions resolve_domain(const std::vector<Attribute> &attributes,
                                     const DomainOptions &requested)
        {
            DomainOptions result = requested;
            if (result.mode == DomainMode::Fixed)
                return result;
            if (!(result.min_segment_width > 0.0))
                throw Error("derived domain requires a positive min_segment_width");
            if (attributes.empty())
                throw Error("cannot derive an attribute domain from no points");

            const auto minmax = std::minmax_element(attributes.begin(), attributes.end());
            const double data_min = static_cast<double>(*minmax.first);
            const double data_max = static_cast<double>(*minmax.second);
            result.attr_min = std::floor(data_min / result.min_segment_width) * result.min_segment_width;
            const double span = data_max - result.attr_min;
            std::uint64_t needed = span > 0.0
                                       ? static_cast<std::uint64_t>(std::ceil(span / result.min_segment_width))
                                       : 1;
            needed = std::max<std::uint64_t>(needed, 1);
            result.num_bins = 1;
            while (result.num_bins < needed)
            {
                if (result.num_bins > 32768)
                    throw Error("derived domain exceeds the 65536-bin limit");
                result.num_bins <<= 1;
            }
            result.attr_max = result.attr_min + result.num_bins * result.min_segment_width;
            while (data_max > result.attr_max)
            {
                if (result.num_bins > 32768)
                    throw Error("derived domain exceeds the 65536-bin limit");
                result.num_bins <<= 1;
                result.attr_max = result.attr_min + result.num_bins * result.min_segment_width;
            }
            return result;
        }
    }

    Index IndexBuilder::build(std::shared_ptr<const DenseDataset> dataset,
                              const std::vector<Attribute> &attributes,
                              const BuildOptions &options)
    {
        if (!dataset)
            throw Error("dataset is null");
        if (attributes.size() != dataset->size())
            throw Error("attribute count does not match dataset point count");
        if (options.M == 0 || options.M > std::numeric_limits<std::uint16_t>::max())
            throw Error("M must be in [1, 65535]");
        if (options.ef_construction == 0)
            throw Error("ef_construction must be positive");
        if (options.threads == 0)
            throw Error("threads must be positive");

        const DomainOptions domain = resolve_domain(attributes, options.domain);
        auto impl = std::make_unique<IndexImpl>(std::move(dataset), options.seed);
        impl->num_points = impl->dataset->size();
        impl->active_points = impl->num_points;
        impl->point_capacity = impl->num_points;
        impl->dimension = impl->dataset->dimension();
        impl->M = options.M;
        impl->ef_construction = options.ef_construction;
        impl->tree.reset(domain.num_bins, domain.attr_min, domain.attr_max);
        impl->degree_by_depth.assign(impl->tree.levels, impl->M);
        impl->initialize_layout();
        impl->initialize_distance();

        impl->attributes.assign(impl->point_capacity, 0);
        std::copy(attributes.begin(), attributes.end(), impl->attributes.begin());
        impl->attribute_keys.assign(impl->point_capacity, 0);
        for (PointId id = 0; id < impl->num_points; ++id)
            impl->attribute_keys[id] = attribute_key(impl->attributes[id]);
        impl->leaf_ids.assign(impl->point_capacity, 0);
        impl->visited.assign(impl->point_capacity, 0);
        impl->degree.assign(static_cast<std::size_t>(impl->point_capacity) * impl->tree.levels, 0);
        impl->edge_records.assign(static_cast<std::size_t>(impl->point_capacity) * impl->capacity_per_point,
                               make_edge_record(kInvalidPointId, 0));

        std::vector<std::uint32_t> bin_counts(impl->tree.num_bins, 0);
        for (PointId id = 0; id < impl->num_points; ++id)
        {
            const std::uint16_t leaf = impl->tree.bin_for_value(attributes[id]);
            impl->leaf_ids[id] = leaf;
            ++bin_counts[leaf];
        }
        impl->tree.clear_members_and_graph_state();
        impl->tree.reserve_members_from_bin_counts(bin_counts);
        for (PointId id = 0; id < impl->num_points; ++id)
            impl->tree.add_member_to_path(impl->leaf_ids[id], id);

        if (options.threads > 1)
        {
            ParallelGraphBuilder builder(*impl);
            builder.build(options.threads);
        }
        else
        {
            SequentialGraphBuilder builder(*impl);
            builder.build();
        }
        impl->build_reverse_adjacency(options.threads);
        return Index(std::move(impl));
    }
}
