#include "index_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <type_traits>
#include <utility>
#include <unistd.h>

namespace rfanns::testing
{
    class IndexAccess
    {
    public:
        static IndexImpl &impl(Index &index) { return *index.impl_; }
        static const IndexImpl &impl(const Index &index) { return *index.impl_; }
    };
}

namespace
{
    template <typename T, typename = void>
    struct has_neighbors_member : std::false_type
    {
    };

    template <typename T>
    struct has_neighbors_member<
        T,
        std::void_t<decltype(std::declval<T &>().neighbors)>> : std::true_type
    {
    };

    static_assert(!has_neighbors_member<rfanns::IndexImpl>::value,
                  "IndexImpl must not retain a duplicate neighbors array");
    static_assert(sizeof(rfanns::EdgeRecord) == 8, "forward edge records must be 64-bit");
    static_assert(sizeof(rfanns::PointId) == 4, "edge destinations must be 32-bit");

    void require(bool condition, const std::string &message)
    {
        if (!condition)
            throw rfanns::Error("test failed: " + message);
    }

    void rewrite_for_fixture(rfanns::IndexImpl &impl,
                             rfanns::PointId owner,
                             std::uint32_t depth,
                             const std::vector<rfanns::PointId> &neighbors)
    {
        impl.commit_neighbor_rewrite_stats(
            impl.write_neighbor_list(owner, depth, neighbors), 0.0);
    }

    std::shared_ptr<rfanns::DenseDataset> make_dataset(std::uint32_t count,
                                                       std::uint32_t dimension = 8)
    {
        std::vector<float> values(static_cast<std::size_t>(count) * dimension);
        for (std::uint32_t id = 0; id < count; ++id)
        {
            for (std::uint32_t component = 0; component < dimension; ++component)
            {
                values[static_cast<std::size_t>(id) * dimension + component] =
                    static_cast<float>((id + 1) * (component + 3) % 97) +
                    static_cast<float>(id) * 0.01f;
            }
        }
        return std::make_shared<rfanns::DenseDataset>(count, dimension, std::move(values));
    }

    void test_squared_l2_reduction()
    {
        constexpr std::size_t dimension = 128;
        std::vector<float> zeros(dimension, 0.0f);
        std::vector<float> sift_maxima(dimension, 241.0f);
        hnswlib::L2Space space(dimension);
        const auto distance_function = space.get_dist_func();
        void *const distance_parameter = space.get_dist_func_param();

        const float sift_distance =
            distance_function(zeros.data(), sift_maxima.data(), distance_parameter);
        const float expected_sift_distance =
            static_cast<float>(dimension * 241U * 241U);
        require(sift_distance == expected_sift_distance,
                "SIMD squared-L2 reduction is exact for SIFT's integral range");

        std::vector<float> left(dimension);
        std::vector<float> right(dimension);
        double reference = 0.0;
        for (std::size_t component = 0; component < dimension; ++component)
        {
            left[component] = static_cast<float>(component) * 0.25f +
                              static_cast<float>(component % 7U) * 0.1f;
            right[component] = static_cast<float>(dimension - component) * 0.125f -
                               static_cast<float>(component % 5U) * 0.2f;
            const double difference = static_cast<double>(left[component]) -
                                      static_cast<double>(right[component]);
            reference += difference * difference;
        }
        const float fractional_distance =
            distance_function(left.data(), right.data(), distance_parameter);
        const double tolerance = 1e-5 * std::max(1.0, reference);
        require(std::abs(static_cast<double>(fractional_distance) - reference) <= tolerance,
                "SIMD squared-L2 reduction agrees with a double-precision reference");
    }

    rfanns::Index make_index(std::uint32_t count = 64)
    {
        const auto data = make_dataset(count);
        std::vector<rfanns::Attribute> attributes(count);
        std::iota(attributes.begin(), attributes.end(), 0);
        rfanns::BuildOptions options;
        options.M = 6;
        options.ef_construction = 24;
        options.threads = 1;
        options.domain.mode = rfanns::DomainMode::Fixed;
        options.domain.attr_min = 0;
        options.domain.attr_max = static_cast<double>(count - 1);
        options.domain.num_bins = 8;
        return rfanns::IndexBuilder::build(data, attributes, options);
    }

    bool slot_is_bound(const rfanns::IndexImpl &impl, rfanns::PointId id)
    {
        return id < impl.num_points &&
               (impl.internal_to_external.empty() ||
                impl.internal_to_external[id] != rfanns::kInvalidPointId);
    }

    std::vector<rfanns::PointId> decoded_forward_destinations(
        const rfanns::IndexImpl &impl)
    {
        std::vector<rfanns::PointId> destinations;
        destinations.reserve(impl.edge_records.size());
        for (const rfanns::EdgeRecord record : impl.edge_records)
            destinations.push_back(rfanns::edge_destination(record));
        return destinations;
    }

    void require_edge_storage_consistency(const rfanns::Index &index,
                                            const std::string &context)
    {
        const auto &impl = rfanns::testing::IndexAccess::impl(index);
        require(impl.edge_records.size() ==
                    static_cast<std::size_t>(impl.point_capacity) * impl.capacity_per_point,
                context + ": forward edge storage has the configured capacity");
        require(impl.attribute_keys.size() == impl.point_capacity,
                context + ": every point slot has an attribute key");
        require(impl.degree.size() == impl.edge_list_capacity_count(),
                context + ": every point-depth pair has a degree");

        for (rfanns::PointId owner = 0; owner < impl.point_capacity; ++owner)
        {
            for (std::uint32_t depth = 0; depth < impl.tree.levels; ++depth)
            {
                const std::uint16_t degree = impl.get_degree(owner, depth);
                require(degree <= impl.capacity_at_depth(depth),
                        context + ": degree fits its depth capacity");
                if (!slot_is_bound(impl, owner))
                    require(degree == 0,
                            context + ": unbound point slots have zero forward degree");

                for (std::uint32_t slot = 0; slot < impl.capacity_at_depth(depth); ++slot)
                {
                    const rfanns::EdgeRecord record =
                        impl.edge_records[impl.neighbor_index(owner, depth, slot)];
                    const rfanns::PointId destination = rfanns::edge_destination(record);
                    const rfanns::AttrKey attribute = rfanns::edge_attribute(record);
                    if (slot < degree)
                    {
                        require(destination < impl.num_points && slot_is_bound(impl, destination),
                                context + ": every live edge targets a bound point");
                        require(attribute == impl.attribute_keys[destination],
                                context + ": edge-record attribute matches its destination");
                    }
                    else
                    {
                        require(destination == rfanns::kInvalidPointId && attribute == 0,
                                context + ": every unused edge slot is canonical invalid");
                    }
                }
            }
        }
    }

    void require_exact_reverse_adjacency(const rfanns::Index &index,
                                         const std::string &context)
    {
        const auto &impl = rfanns::testing::IndexAccess::impl(index);
        require_edge_storage_consistency(index, context);
        require(impl.reverse_adjacency_ready, context + ": reverse adjacency is ready");
        require(impl.reverse_adjacency.size() == impl.edge_list_capacity_count(),
                context + ": reverse adjacency has one list per point and depth");

        std::vector<std::vector<rfanns::PointId>> expected(impl.edge_list_capacity_count());
        std::uint64_t edge_count = 0;
        for (rfanns::PointId owner = 0; owner < impl.num_points; ++owner)
        {
            if (!slot_is_bound(impl, owner))
                continue;
            for (std::uint32_t depth = 0; depth < impl.tree.levels; ++depth)
            {
                for (std::uint16_t slot = 0; slot < impl.get_degree(owner, depth); ++slot)
                {
                    const rfanns::PointId target = impl.get_neighbor(owner, depth, slot);
                    require(slot_is_bound(impl, target),
                            context + ": every forward target is bound and in range");
                    expected[impl.edge_list_index(target, depth)].push_back(owner);
                    ++edge_count;
                }
            }
        }

        for (std::size_t list_id = 0; list_id < expected.size(); ++list_id)
        {
            auto wanted = expected[list_id];
            auto actual = impl.reverse_adjacency[list_id];
            std::sort(wanted.begin(), wanted.end());
            std::sort(actual.begin(), actual.end());
            require(std::adjacent_find(actual.begin(), actual.end()) == actual.end(),
                    context + ": reverse list has no duplicate owner");
            require(actual == wanted,
                    context + ": reverse list has exactly the forward-edge owners");
        }
        require(impl.stats.reverse_adjacency_edges == edge_count,
                context + ": reverse-edge statistic equals the physical edge count");
    }

    void require_same_topology(const rfanns::Index &left,
                               const rfanns::Index &right,
                               const std::string &context)
    {
        const auto &a = rfanns::testing::IndexAccess::impl(left);
        const auto &b = rfanns::testing::IndexAccess::impl(right);
        require(a.num_points == b.num_points && a.point_capacity == b.point_capacity &&
                    a.tree.levels == b.tree.levels &&
                    a.degree_by_depth == b.degree_by_depth,
                context + ": graph layouts match");
        require(a.degree == b.degree, context + ": degrees match");
        require(decoded_forward_destinations(a) == decoded_forward_destinations(b),
                context + ": ordered decoded neighbor slots match");
        require_edge_storage_consistency(left, context + " left");
        require_edge_storage_consistency(right, context + " right");
        require(a.tree.nodes.size() == b.tree.nodes.size(), context + ": tree shapes match");
        for (std::size_t node_id = 0; node_id < a.tree.nodes.size(); ++node_id)
        {
            const auto &x = a.tree.nodes[node_id];
            const auto &y = b.tree.nodes[node_id];
            require(x.members == y.members,
                    context + ": ordered members match");
        }
    }

    void require_sorted_reverse_owners(const rfanns::Index &index,
                                       const std::string &context)
    {
        const auto &impl = rfanns::testing::IndexAccess::impl(index);
        require(impl.reverse_adjacency_ready, context + ": reverse adjacency is ready");
        for (const auto &owners : impl.reverse_adjacency)
            require(std::is_sorted(owners.begin(), owners.end()),
                    context + ": reverse owners use deterministic ascending order");
    }

    void test_build_time_reverse_adjacency()
    {
        rfanns::Index index = make_index();
        require_exact_reverse_adjacency(index, "fresh static build");
        require_sorted_reverse_owners(index, "fresh static build");
        index.validate();
    }

    void test_hnsw_neighbor_heuristic()
    {
        auto data = std::make_shared<rfanns::DenseDataset>(
            4,
            2,
            std::vector<float>{0.0f, 0.0f,
                               1.0f, 0.0f,
                               1.1f, 0.0f,
                               -1.2f, 0.0f});
        const std::vector<rfanns::Attribute> attributes(4, 0);
        rfanns::BuildOptions options;
        options.M = 3;
        options.ef_construction = 4;
        options.threads = 1;
        options.domain.mode = rfanns::DomainMode::Fixed;
        options.domain.attr_min = 0;
        options.domain.attr_max = 1;
        options.domain.num_bins = 1;
        rfanns::Index index = rfanns::IndexBuilder::build(data, attributes, options);
        const auto &impl = rfanns::testing::IndexAccess::impl(index);

        const auto diversified = impl.select_hnsw_neighbors(0, {2, 3, 1, 2}, 2);
        require(diversified == std::vector<rfanns::PointId>({1, 3}),
                "HNSW heuristic rejects a candidate shadowed by a closer selected point");

        const std::vector<rfanns::IndexImpl::DistanceId> scored = {
            {impl.distance(0, 2), 2},
            {impl.distance(0, 3), 3},
            {impl.distance(0, 1), 1},
            {impl.distance(0, 2), 2}};
        const auto diversified_scored = impl.select_hnsw_neighbors_scored(0, scored, 2);
        require(diversified_scored == diversified,
                "known-distance HNSW selection matches the reference selector exactly");

        const auto below_capacity = impl.select_hnsw_neighbors(0, {2, 1}, 3);
        require(below_capacity == std::vector<rfanns::PointId>({1, 2}),
                "HNSW heuristic preserves all distance-ordered candidates below capacity");

        const std::vector<rfanns::IndexImpl::DistanceId> below_capacity_scored = {
            {impl.distance(0, 2), 2}, {impl.distance(0, 1), 1}};
        require(impl.select_hnsw_neighbors_scored(0, below_capacity_scored, 3) ==
                    below_capacity,
                "known-distance selector preserves ordered candidates below capacity");
    }

    void test_static_dynamic_topology_equivalence()
    {
        constexpr std::uint32_t count = 96;
        constexpr std::uint32_t prefix = 21;
        const auto full_data = make_dataset(count);
        std::vector<rfanns::Attribute> full_attributes(count);
        std::iota(full_attributes.begin(), full_attributes.end(), 0);

        const auto options = [&]() {
            rfanns::BuildOptions value;
            value.M = 8;
            value.ef_construction = 32;
            value.threads = 1;
            value.domain.mode = rfanns::DomainMode::Fixed;
            value.domain.attr_min = 0;
            value.domain.attr_max = count - 1;
            value.domain.num_bins = 8;
            return value;
        }();

        rfanns::Index full = rfanns::IndexBuilder::build(full_data, full_attributes, options);
        std::vector<float> prefix_values(full_data->values().begin(),
                                         full_data->values().begin() +
                                             static_cast<std::size_t>(prefix) *
                                                 full_data->dimension());
        auto prefix_data = std::make_shared<rfanns::DenseDataset>(
            prefix, full_data->dimension(), std::move(prefix_values));
        std::vector<rfanns::Attribute> prefix_attributes(full_attributes.begin(),
                                                         full_attributes.begin() + prefix);
        rfanns::Index incremental =
            rfanns::IndexBuilder::build(prefix_data, prefix_attributes, options);
        incremental.reserve(count);
        for (rfanns::PointId id = prefix; id < count; ++id)
        {
            const rfanns::PointId inserted =
                incremental.insert(id, full_data->row(id), full_attributes[id]);
            require(inserted == id, "dynamic completion preserves the static internal-ID order");
        }

        full.validate();
        incremental.validate();
        require_same_topology(full,
                              incremental,
                              "single-thread static build and prefix plus dynamic inserts");
        require_exact_reverse_adjacency(full, "complete static build");
        require_exact_reverse_adjacency(incremental, "dynamic completion");
    }

    void test_leaf_to_root_seed_propagation()
    {
        const auto data = make_dataset(3);
        std::vector<float> prefix_values(data->values().begin(),
                                         data->values().begin() +
                                             2 * data->dimension());
        auto prefix_data = std::make_shared<rfanns::DenseDataset>(
            2, data->dimension(), std::move(prefix_values));
        const std::vector<rfanns::Attribute> attributes = {0, 3};
        rfanns::BuildOptions options;
        options.M = 2;
        options.ef_construction = 1;
        options.threads = 1;
        options.domain.mode = rfanns::DomainMode::Fixed;
        options.domain.attr_min = 0;
        options.domain.attr_max = 4;
        options.domain.num_bins = 4;
        rfanns::Index index = rfanns::IndexBuilder::build(prefix_data, attributes, options);
        index.reserve(3);

        auto &impl = rfanns::testing::IndexAccess::impl(index);
        require(impl.tree.levels == 3, "routing fixture has leaf, parent, and root graphs");
        rewrite_for_fixture(impl, 0, 0, {});
        rewrite_for_fixture(impl, 1, 0, {});
        require_exact_reverse_adjacency(index, "routing fixture after root disconnection");

        const rfanns::PointId inserted = index.insert(2, data->row(2), 3);
        require(inserted == 2, "routing fixture appends the new point");
        require(impl.get_degree(inserted, 0) == 1 &&
                    impl.get_neighbor(inserted, 0, 0) == 1,
                "leaf candidate is propagated upward and overrides the disconnected root seed");
        require_exact_reverse_adjacency(index, "leaf-to-root routed insert");
        index.validate();
    }

    void test_inherited_region_frontier()
    {
        const auto data = make_dataset(5);
        const std::vector<rfanns::Attribute> attributes = {0, 1, 4, 5, 1};
        rfanns::BuildOptions options;
        options.M = 4;
        options.ef_construction = 8;
        options.threads = 1;
        options.domain.mode = rfanns::DomainMode::Fixed;
        options.domain.attr_min = 0;
        options.domain.attr_max = 7;
        options.domain.num_bins = 4;
        rfanns::Index index = rfanns::IndexBuilder::build(data, attributes, options);
        auto &impl = rfanns::testing::IndexAccess::impl(index);

        constexpr std::uint32_t root_depth = 0;
        const std::int32_t root = impl.tree.path_by_leaf[impl.leaf_ids[0]][root_depth];
        const std::int32_t inherited_child = impl.tree.path_by_leaf[impl.leaf_ids[0]][1];
        require(root == 0 && inherited_child != root,
                "inherited-frontier fixture has a child-to-parent route");

        rewrite_for_fixture(impl, 0, root_depth, {1, 2});
        rewrite_for_fixture(impl, 1, root_depth, {});
        rewrite_for_fixture(impl, 2, root_depth, {3});
        rewrite_for_fixture(impl, 3, root_depth, {});
        rewrite_for_fixture(impl, 4, root_depth, {});

        const std::vector<rfanns::IndexImpl::DistanceId> inherited = {
            {impl.distance(4, 0), 0}};
        std::vector<rfanns::IndexImpl::DistanceId> candidate_distances;
        impl.search_insert_candidates(root,
                                      4,
                                      root_depth,
                                      0,
                                      inherited_child,
                                      inherited,
                                      candidate_distances);

        const auto contains = [&](rfanns::PointId id) {
            return std::find_if(candidate_distances.begin(),
                                candidate_distances.end(),
                                [&](const rfanns::IndexImpl::DistanceId &candidate) {
                                    return candidate.second == id;
                                }) != candidate_distances.end();
        };
        require(contains(0), "the complete inherited frontier remains eligible at the parent");
        require(!contains(1), "an unseen neighbor inside the inherited child is not rescored");
        require(contains(2) && contains(3),
                "parent traversal explores the newly added region beyond its first discovered point");
        require(impl.visited[1] == impl.visited_tag,
                "an ignored child-side neighbor is still marked visited before interval pruning");
        const auto added_region_candidate = std::find_if(
            candidate_distances.begin(),
            candidate_distances.end(),
            [](const rfanns::IndexImpl::DistanceId &candidate) { return candidate.second == 2; });
        require(added_region_candidate != candidate_distances.end() &&
                    added_region_candidate->first == impl.distance(4, 2),
                "a newly discovered candidate carries its computed query distance");
    }

    void test_point_parallel_build()
    {
        constexpr std::uint32_t count = 256;
        const auto data = make_dataset(count);
        std::vector<rfanns::Attribute> attributes(count);
        std::iota(attributes.begin(), attributes.end(), 0);

        const auto build = [&](std::uint32_t threads) {
            rfanns::BuildOptions options;
            options.M = 8;
            options.ef_construction = 32;
            options.threads = threads;
            options.domain.mode = rfanns::DomainMode::Fixed;
            options.domain.attr_min = 0;
            options.domain.attr_max = count - 1;
            options.domain.num_bins = 8;
            return rfanns::IndexBuilder::build(data, attributes, options);
        };

        rfanns::Index point_sequential = build(1);
        require_edge_storage_consistency(point_sequential, "sequential point build");
        point_sequential.validate();
        require_exact_reverse_adjacency(point_sequential, "sequential point build");

        rfanns::Index point_parallel = build(4);
        point_parallel.validate();
        require_exact_reverse_adjacency(point_parallel, "point-parallel build");
        require_sorted_reverse_owners(point_parallel, "point-parallel build");
        point_parallel.prepare_search();
        const auto result = point_parallel.search(data->row(100),
                                                  100,
                                                  {32, 180},
                                                  rfanns::SearchOptions{10, 64});
        require(!result.empty(), "point-parallel index is searchable");
        for (const auto &item : result)
            require(item.id >= 32 && item.id <= 180,
                    "point-parallel search obeys the attribute range");
        point_parallel.prepare_updates();
        require(point_parallel.erase(100), "point-parallel index supports physical delete");
        point_parallel.validate();
        require_exact_reverse_adjacency(point_parallel,
                                        "point-parallel build after physical delete");
    }

    std::vector<char> read_bytes(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    template <typename T>
    void write_scalar(std::ofstream &output, const T &value)
    {
        output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    void test_range_tree()
    {
        rfanns::RangeTree tree(8, 0.0, 80.0);
        require(tree.levels == 4 && tree.nodes.size() == 15, "complete tree shape");
        require(tree.bin_for_value(0) == 0, "minimum maps to first bin");
        require(tree.bin_for_value(80) == 7, "inclusive maximum maps to last bin");
        require(tree.bin_for_value(10) == 1, "internal boundary maps to right bin");
        std::uint16_t left = 0;
        std::uint16_t right = 0;
        require(tree.query_bins(-10, 25, left, right) && left == 0 && right == 2,
                "query clipping and bin conversion");
        const auto cover = tree.canonical_cover(1, 6);
        require(cover.size() == 4, "canonical cover size");
        std::uint32_t covered = 0;
        for (const auto node_id : cover)
            covered += tree.nodes[node_id].leaf_r - tree.nodes[node_id].leaf_l + 1;
        require(covered == 6, "canonical cover has no overlap or gaps");
    }

    void test_attribute_keys()
    {
        const std::array<rfanns::Attribute, 5> values = {
            std::numeric_limits<rfanns::Attribute>::min(), -1, 0, 1,
            std::numeric_limits<rfanns::Attribute>::max()};
        for (std::size_t index = 1; index < values.size(); ++index)
            require(rfanns::attribute_key(values[index - 1]) < rfanns::attribute_key(values[index]),
                    "attribute keys preserve signed int32 order");
        const auto record = rfanns::make_edge_record(17, rfanns::attribute_key(-1));
        require(rfanns::edge_destination(record) == 17 &&
                    rfanns::edge_attribute(record) == rfanns::attribute_key(-1),
                "edge-record roundtrip");
    }

    void test_derived_domain()
    {
        const auto data = make_dataset(4);
        const std::vector<rfanns::Attribute> attributes = {3, 4, 8, 12};
        rfanns::BuildOptions options;
        options.M = 2;
        options.ef_construction = 4;
        options.threads = 1;
        options.domain.mode = rfanns::DomainMode::Derived;
        options.domain.min_segment_width = 2.0;
        rfanns::Index index = rfanns::IndexBuilder::build(data, attributes, options);
        const auto metadata = index.metadata();
        require(metadata.num_bins == 8 && metadata.attr_min == 2.0 && metadata.attr_max == 18.0,
                "derived domain is the smallest aligned power-of-two range");
        index.validate();
    }

    void test_cross_depth_edge_deduplication()
    {
        const auto data = make_dataset(4);
        const std::vector<rfanns::Attribute> attributes = {0, 1, 2, 6};
        rfanns::BuildOptions options;
        options.M = 3;
        options.ef_construction = 6;
        options.threads = 1;
        options.domain.mode = rfanns::DomainMode::Fixed;
        options.domain.attr_min = 0;
        options.domain.attr_max = 7;
        options.domain.num_bins = 4;
        rfanns::Index index = rfanns::IndexBuilder::build(data, attributes, options);
        index.prepare_search();
        auto &impl = rfanns::testing::IndexAccess::impl(index);
        rewrite_for_fixture(impl, 0, 0, {1});
        rewrite_for_fixture(impl, 0, 1, {1, 2});
        rewrite_for_fixture(impl, 0, 2, {});
        require(impl.edge_scope_low[impl.edge_list_index(0, 1)] ==
                        rfanns::attribute_key(1) &&
                    impl.edge_scope_high[impl.edge_list_index(0, 1)] ==
                        rfanns::attribute_key(2),
                "edge scopes use the signed-order-preserving int32 key");
        require(impl.edge_scope_low[impl.edge_list_index(0, 2)] ==
                        std::numeric_limits<rfanns::AttrKey>::max() &&
                    impl.edge_scope_high[impl.edge_list_index(0, 2)] == 0,
                "an empty edge scope uses the low-greater-than-high sentinel");

        impl.begin_cut_cache_query();
        const std::uint32_t tag = impl.next_visited_tag();
        std::vector<rfanns::PointId> selected;
        impl.select_edges(0,
                          0,
                          2,
                          rfanns::attribute_key(0),
                          rfanns::attribute_key(5),
                          2,
                          tag,
                          selected);
        require(selected == std::vector<rfanns::PointId>({1, 2}),
                "a duplicate neighbor across depths does not consume the edge limit");
        require(impl.visited[1] == tag && impl.visited[2] == tag,
                "cross-depth edge selection marks each accepted neighbor exactly once");
    }

    void test_static_search_and_v1_roundtrip(const std::filesystem::path &directory)
    {
        rfanns::Index index = make_index();
        index.validate();
        const auto data = make_dataset(64);
        index.prepare_search();
        const auto result = index.search(data->row(17),
                                         3,
                                         {8, 30},
                                         rfanns::SearchOptions{10, 40});
        require(!result.empty(), "search returns candidates");
        for (const auto &item : result)
            require(item.id >= 8 && item.id <= 30, "search result obeys exact range");

        const auto first = directory / "roundtrip_v1.bin";
        const auto second = directory / "roundtrip_v1_resaved.bin";
        index.save(first.string());
        const std::vector<char> first_bytes = read_bytes(first);
        require(!first_bytes.empty(), "static topology serializes to a nonempty index");
        const std::array<char, 8> expected_magic = {
            'R', 'S', 'H', 'N', 'S', 'W', 'E', '\0'};
        require(first_bytes.size() >= expected_magic.size() &&
                    std::equal(expected_magic.begin(), expected_magic.end(), first_bytes.begin()),
                "v1 serialization retains the RSHNSWE compatibility identifier");
        rfanns::Index loaded = rfanns::Index::load(first.string(), data);
        require_edge_storage_consistency(loaded, "v1 immediately after load");
        loaded.save(second.string());
        require(read_bytes(first) == read_bytes(second), "v1 load/save is byte stable");
        const auto &loaded_impl = rfanns::testing::IndexAccess::impl(loaded);
        require(!loaded_impl.reverse_adjacency_ready,
                "v1 load does not deserialize an unavailable reverse graph");
        loaded.prepare_search();
        require(!loaded_impl.reverse_adjacency_ready,
                "prepare_search builds query metadata without reverse adjacency");
        const auto loaded_result = loaded.search(data->row(17),
                                                 3,
                                                 {8, 30},
                                                 rfanns::SearchOptions{10, 40});
        require(result.size() == loaded_result.size(), "reloaded search result size");
        for (std::size_t i = 0; i < result.size(); ++i)
            require(result[i].id == loaded_result[i].id && result[i].distance == loaded_result[i].distance,
                    "reloaded search result equality");
        loaded.prepare_updates();
        require_exact_reverse_adjacency(loaded, "v1 after explicit update preparation");
    }

    void test_invalid_magic_rejection(const std::filesystem::path &directory)
    {
        const auto valid = directory / "valid_magic.bin";
        const auto invalid = directory / "invalid_magic.bin";
        const auto data = make_dataset(64);
        rfanns::Index index = make_index();
        index.save(valid.string());
        std::vector<char> bytes = read_bytes(valid);
        require(bytes.size() >= rfanns::kIndexFileMagic.size(),
                "saved index contains the complete file magic");
        bytes.front() = 'X';
        std::ofstream output(invalid, std::ios::binary);
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        output.close();

        bool rejected = false;
        try
        {
            (void)rfanns::Index::load(invalid.string(), data);
        }
        catch (const rfanns::Error &error)
        {
            rejected = std::string(error.what()).find("invalid RFANNS index magic") !=
                       std::string::npos;
        }
        require(rejected, "an index with a different file magic is rejected");
    }

    void write_minimal_v2(const std::filesystem::path &path,
                          std::uint32_t serialized_graph_size = 1)
    {
        std::ofstream output(path, std::ios::binary);
        output.write(rfanns::kIndexFileMagic.data(),
                     static_cast<std::streamsize>(rfanns::kIndexFileMagic.size()));
        const std::uint32_t version = 2;
        const std::uint32_t one = 1;
        const std::uint32_t dimension = 2;
        const std::uint32_t M = 4;
        const std::uint32_t ef = 20;
        const double attr_min = 0;
        const double attr_max = 10;
        const std::uint32_t node_count = 1;
        const std::uint32_t depth_capacity = 3;
        write_scalar(output, version);
        write_scalar(output, one);
        write_scalar(output, dimension);
        write_scalar(output, one);
        write_scalar(output, one);
        write_scalar(output, M);
        write_scalar(output, ef);
        write_scalar(output, attr_min);
        write_scalar(output, attr_max);
        write_scalar(output, node_count);
        write_scalar(output, depth_capacity);
        const std::array<std::int32_t, 7> node = {0, 0, 0, 0, -1, -1, -1};
        for (const auto value : node)
            write_scalar(output, value);
        const double attribute = 5;
        const std::uint16_t leaf = 0;
        const std::uint16_t degree = 0;
        write_scalar(output, serialized_graph_size);
        write_scalar(output, attribute);
        write_scalar(output, leaf);
        write_scalar(output, degree);
        for (std::uint32_t i = 0; i < depth_capacity; ++i)
            write_scalar(output, rfanns::kInvalidPointId);
    }

    void test_v2_roundtrip_and_rank_rejection(const std::filesystem::path &directory)
    {
        const auto source = directory / "minimal_v2.bin";
        const auto saved = directory / "minimal_v2_saved.bin";
        write_minimal_v2(source);
        auto data = std::make_shared<rfanns::DenseDataset>(
            1, 2, std::vector<float>{1.0f, 2.0f});
        rfanns::Index index = rfanns::Index::load(source.string(), data);
        require_edge_storage_consistency(index, "v2 immediately after load");
        index.validate();
        index.save(saved.string());
        require(read_bytes(source) == read_bytes(saved), "v2 load/save is byte stable");

        const auto inconsistent = directory / "minimal_v2_bad_graph_size.bin";
        write_minimal_v2(inconsistent, 0);
        bool graph_size_rejected = false;
        try
        {
            (void)rfanns::Index::load(inconsistent.string(), data);
        }
        catch (const rfanns::Error &error)
        {
            graph_size_rejected =
                std::string(error.what()).find("graph size") != std::string::npos;
        }
        require(graph_size_rejected,
                "legacy graph-size field is checked against reconstructed members on load");

        for (const std::uint32_t version : {4U, 5U})
        {
            const auto rank = directory / ("rank_v" + std::to_string(version) + ".bin");
            std::ofstream output(rank, std::ios::binary);
            output.write(rfanns::kIndexFileMagic.data(),
                         static_cast<std::streamsize>(rfanns::kIndexFileMagic.size()));
            write_scalar(output, version);
            output.close();
            bool rejected = false;
            try
            {
                (void)rfanns::Index::load(rank.string(), data);
            }
            catch (const rfanns::Error &error)
            {
                rejected = std::string(error.what()).find("unsupported") != std::string::npos;
            }
            require(rejected, "rank_bins v4/v5 is rejected explicitly");
        }
    }

    void test_insert_and_expansion(const std::filesystem::path &directory)
    {
        rfanns::Index index = make_index();
        const auto full_data = make_dataset(80);
        index.reserve(80);
        index.prepare_search();
        const rfanns::PointId inserted = index.insert(64, full_data->row(64), 32);
        require(inserted == 64 && index.metadata().num_points == 65, "in-domain append insert");
        require_edge_storage_consistency(index, "after in-domain insert");
        index.validate();

        auto &impl = rfanns::testing::IndexAccess::impl(index);
        const auto old_degree = impl.degree;
        const auto old_neighbors = decoded_forward_destinations(impl);
        const auto old_offsets = impl.depth_offsets;
        const auto old_caps = impl.degree_by_depth;
        const std::uint32_t old_levels = impl.tree.levels;
        const std::uint32_t old_capacity = impl.capacity_per_point;
        impl.expand_domain_to_include(100);
        require_edge_storage_consistency(index, "after right domain expansion");
        require(impl.tree.attr_min == 0 && impl.tree.attr_max == 126,
                "right expansion doubles domain");
        require(impl.tree.num_bins == 16 && impl.tree.levels == old_levels + 1,
                "right expansion doubles bins and adds a level");
        for (rfanns::PointId id = 0; id < 65; ++id)
        {
            require(impl.get_degree(id, 0) == old_degree[static_cast<std::size_t>(id) * old_levels],
                    "old root degree copied to new root");
            require(impl.get_degree(id, 1) == old_degree[static_cast<std::size_t>(id) * old_levels],
                    "old root degree shifted to old-root child");
            for (std::uint16_t slot = 0; slot < impl.get_degree(id, 0); ++slot)
            {
                const auto old_neighbor = old_neighbors[static_cast<std::size_t>(id) * old_capacity +
                                                        old_offsets[0] + slot];
                require(impl.get_neighbor(id, 0, slot) == old_neighbor &&
                            impl.get_neighbor(id, 1, slot) == old_neighbor,
                        "old root neighbors copied at both depths");
            }
        }
        index.insert(65, full_data->row(65), 100, rfanns::InsertOptions{true});
        require_edge_storage_consistency(index, "after expansion-triggering insert");
        index.validate();

        const auto expanded = directory / "expanded.bin";
        const auto id_map = directory / "expanded.idmap";
        index.save(expanded.string(), id_map.string());
        rfanns::Index loaded = rfanns::Index::load(expanded.string(), full_data, id_map.string(), true);
        require_edge_storage_consistency(loaded, "expanded index immediately after load");
        loaded.prepare_search();
        loaded.validate();
        const auto before = index.search(full_data->row(20),
                                         7,
                                         {0, 110},
                                         rfanns::SearchOptions{10, 40});
        const auto after = loaded.search(full_data->row(20),
                                         7,
                                         {0, 110},
                                         rfanns::SearchOptions{10, 40});
        require(before.size() == after.size(), "expanded reload result size");
        for (std::size_t i = 0; i < before.size(); ++i)
            require(before[i].id == after[i].id, "expanded reload search IDs");

        rfanns::Index left = make_index();
        left.reserve(80);
        left.prepare_search();
        auto &left_impl = rfanns::testing::IndexAccess::impl(left);
        left_impl.expand_domain_to_include(-1);
        require_edge_storage_consistency(left, "after left domain expansion");
        require(left_impl.tree.attr_min == -63 && left_impl.tree.attr_max == 63,
                "left expansion doubles domain");
        require(left_impl.leaf_ids[0] >= 8, "left expansion offsets old leaves");
        left_impl.expand_domain_to_include(-500);
        require_edge_storage_consistency(left, "after repeated left domain expansion");
        require(left_impl.tree.attr_min <= -500 && left_impl.stats.attr_expand_count > 1,
                "multiple left expansions");

        rfanns::Index disabled = make_index();
        disabled.reserve(80);
        bool threw = false;
        try
        {
            disabled.insert(64, full_data->row(64), 100, rfanns::InsertOptions{false});
        }
        catch (const rfanns::Error &)
        {
            threw = true;
        }
        require(threw, "out-of-domain insert fails when expansion is disabled");

        rfanns::Index limited = make_index();
        auto &limited_impl = rfanns::testing::IndexAccess::impl(limited);
        limited_impl.tree.num_bins = 65536;
        bool limit_rejected = false;
        try
        {
            limited_impl.expand_domain_to_include(100);
        }
        catch (const rfanns::Error &)
        {
            limit_rejected = true;
        }
        require(limit_rejected, "dynamic expansion rejects more than 65536 bins");
    }

    void test_delete_and_reuse(const std::filesystem::path &directory)
    {
        rfanns::Index sequential = make_index();
        sequential.reserve(80);
        sequential.prepare_updates();
        require(sequential.erase(10), "sequential delete succeeds");
        require_edge_storage_consistency(sequential, "after sequential delete");
        sequential.validate();

        const auto &a = rfanns::testing::IndexAccess::impl(sequential);
        require(!a.internal_to_external.empty() && !slot_is_bound(a, 10) &&
                    a.internal_to_external[10] == rfanns::kInvalidPointId,
                "delete materializes the ID map and unbinds the deleted slot");
        require(a.to_internal(10) == rfanns::kInvalidPointId,
                "the removed external ID no longer resolves through the ID map");
        for (std::uint32_t depth = 0; depth < a.tree.levels; ++depth)
            require(a.reverse_adjacency[a.edge_list_index(10, depth)].empty(),
                    "physical delete leaves every exact reverse list for the slot empty");
        for (rfanns::PointId owner = 0; owner < a.num_points; ++owner)
        {
            if (!slot_is_bound(a, owner))
                continue;
            for (std::uint32_t depth = 0; depth < a.tree.levels; ++depth)
            {
                for (std::uint16_t slot = 0; slot < a.get_degree(owner, depth); ++slot)
                    require(a.get_neighbor(owner, depth, slot) != 10,
                            "deleted point has no incoming edge");
            }
        }

        bool save_rejected = false;
        try
        {
            sequential.save((directory / "deleted.bin").string());
        }
        catch (const rfanns::Error &)
        {
            save_rejected = true;
        }
        require(save_rejected, "deleted state cannot be serialized as v1/v2");

        const auto full_data = make_dataset(80);
        const rfanns::PointId reused = sequential.insert(70, full_data->row(70), 20);
        require(reused == 10, "insert reuses the deleted internal ID");
        require(slot_is_bound(a, reused) && a.to_internal(70) == reused &&
                    a.to_internal(10) == rfanns::kInvalidPointId,
                "reused-slot liveness comes solely from the replacement ID-map binding");
        require_edge_storage_consistency(sequential, "after deleted-slot reuse");
        sequential.validate();
        require_exact_reverse_adjacency(sequential, "reused slot before expansion");

        auto &mutable_impl = rfanns::testing::IndexAccess::impl(sequential);
        mutable_impl.expand_domain_to_include(-100);
        require(mutable_impl.internal_to_external[reused] == 70,
                "left domain expansion preserves the reused slot binding");
        require_edge_storage_consistency(sequential,
                                           "reused slot after left domain expansion");
        sequential.validate();
        require_exact_reverse_adjacency(sequential, "reused slot after left expansion");
        mutable_impl.expand_domain_to_include(300);
        require(mutable_impl.internal_to_external[reused] == 70,
                "right domain expansion preserves the reused slot binding");
        require_edge_storage_consistency(sequential,
                                           "reused slot after right domain expansion");
        sequential.validate();
        require_exact_reverse_adjacency(sequential, "reused slot after right expansion");

        const auto reused_index = directory / "reused.bin";
        const auto reused_map = directory / "reused.idmap";
        sequential.save(reused_index.string(), reused_map.string());
        rfanns::Index reloaded = rfanns::Index::load(
            reused_index.string(), full_data, reused_map.string(), true);
        require_edge_storage_consistency(reloaded, "reused index immediately after load");
        reloaded.validate();
        require(reloaded.current_id_map().at(10) == 70,
                "non-identity reused ID map survives save/load");
        reloaded.prepare_updates();
        require_exact_reverse_adjacency(reloaded,
                                        "expanded reused-slot index after save/load");
    }
}

int main()
{
    try
    {
        const auto directory =
            std::filesystem::temp_directory_path() /
            ("rfanns_tests_" + std::to_string(::getpid()));
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        test_squared_l2_reduction();
        test_range_tree();
        test_attribute_keys();
        test_derived_domain();
        test_cross_depth_edge_deduplication();
        test_build_time_reverse_adjacency();
        test_hnsw_neighbor_heuristic();
        test_static_dynamic_topology_equivalence();
        test_leaf_to_root_seed_propagation();
        test_inherited_region_frontier();
        test_point_parallel_build();
        test_static_search_and_v1_roundtrip(directory);
        test_invalid_magic_rejection(directory);
        test_v2_roundtrip_and_rank_rejection(directory);
        test_insert_and_expansion(directory);
        test_delete_and_reuse(directory);
        std::filesystem::remove_all(directory);
        std::cout << "all RFANNS tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
