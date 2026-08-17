#include "index_internal.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace rfanns
{
    namespace
    {
        template <typename T>
        void read_scalar(std::ifstream &input, T &value, const std::string &name)
        {
            input.read(reinterpret_cast<char *>(&value), sizeof(value));
            if (!input)
                throw Error("index ended while reading " + name);
        }

        template <typename T>
        void read_array(std::ifstream &input,
                        std::vector<T> &values,
                        std::size_t count,
                        const std::string &name)
        {
            values.resize(count);
            input.read(reinterpret_cast<char *>(values.data()),
                       static_cast<std::streamsize>(values.size() * sizeof(T)));
            if (!input)
                throw Error("index ended while reading " + name);
        }

        template <typename T>
        void write_scalar(std::ofstream &output, const T &value)
        {
            output.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }

        constexpr std::size_t kEdgeIoChunk = 1U << 20;

        void read_edges(std::ifstream &input,
                        std::vector<EdgeRecord> &edge_records,
                        std::size_t count,
                        const std::vector<AttrKey> &attributes,
                        std::uint32_t num_points)
        {
            edge_records.resize(count, make_edge_record(kInvalidPointId, 0));
            if (count == 0)
                return;
            std::vector<PointId> destinations(std::min(count, kEdgeIoChunk));
            for (std::size_t offset = 0; offset < count; offset += destinations.size())
            {
                const std::size_t chunk = std::min(destinations.size(), count - offset);
                input.read(reinterpret_cast<char *>(destinations.data()),
                           static_cast<std::streamsize>(chunk * sizeof(PointId)));
                if (!input)
                    throw Error("index ended while reading neighbor array");
                for (std::size_t index = 0; index < chunk; ++index)
                {
                    const PointId destination = destinations[index];
                    edge_records[offset + index] =
                        destination == kInvalidPointId || destination >= num_points
                            ? make_edge_record(destination, 0)
                            : make_edge_record(destination, attributes[destination]);
                }
            }
        }

        void write_edge_destinations(std::ofstream &output,
                                     const std::vector<EdgeRecord> &edge_records,
                                     std::size_t count)
        {
            if (count == 0)
                return;
            std::vector<PointId> destinations(std::min(count, kEdgeIoChunk));
            for (std::size_t offset = 0; offset < count; offset += destinations.size())
            {
                const std::size_t chunk = std::min(destinations.size(), count - offset);
                for (std::size_t index = 0; index < chunk; ++index)
                    destinations[index] = edge_destination(edge_records[offset + index]);
                output.write(reinterpret_cast<const char *>(destinations.data()),
                             static_cast<std::streamsize>(chunk * sizeof(PointId)));
                if (!output)
                    throw Error("failed while writing neighbor array");
            }
        }

        bool uniform_degree(const IndexImpl &index)
        {
            return std::all_of(index.degree_by_depth.begin(),
                               index.degree_by_depth.end(),
                               [&](std::uint32_t value) { return value == index.M; });
        }
    }

    std::unique_ptr<IndexImpl> IndexImpl::load(const std::string &path,
                                               std::shared_ptr<const DenseDataset> dataset,
                                               std::uint64_t seed,
                                               bool allow_data_superset)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw Error("cannot open index " + path);

        char magic[8] = {};
        input.read(magic, sizeof(magic));
        if (!input || std::memcmp(magic, kIndexFileMagic.data(), sizeof(magic)) != 0)
            throw Error("invalid RFANNS index magic");

        std::uint32_t version = 0;
        read_scalar(input, version, "version");
        if (version == 4 || version == 5)
            throw Error("rank-bin index versions 4 and 5 are unsupported");
        if (version != 1 && version != 2)
            throw Error("unsupported RFANNS index version; expected 1 or 2");

        auto index = std::make_unique<IndexImpl>(std::move(dataset), seed);
        std::uint32_t num_bins = 0;
        std::uint32_t levels = 0;
        std::uint32_t tree_node_count = 0;
        double attr_min = 0.0;
        double attr_max = 0.0;
        read_scalar(input, index->num_points, "num_points");
        read_scalar(input, index->dimension, "dimension");
        read_scalar(input, num_bins, "num_bins");
        read_scalar(input, levels, "levels");
        read_scalar(input, index->M, "M");
        read_scalar(input, index->ef_construction, "ef_construction");
        read_scalar(input, attr_min, "attr_min");
        read_scalar(input, attr_max, "attr_max");
        read_scalar(input, tree_node_count, "tree_node_count");

        if (index->num_points == 0 || index->dimension == 0 || index->M == 0)
            throw Error("index contains invalid zero-valued metadata");
        if (index->dataset->dimension() != index->dimension)
            throw Error("dataset dimension does not match index metadata");
        if (allow_data_superset)
        {
            if (index->dataset->size() < index->num_points)
                throw Error("dataset has fewer points than the index");
        }
        else if (index->dataset->size() != index->num_points)
        {
            throw Error("dataset point count does not match index metadata");
        }

        index->tree.reset(num_bins, attr_min, attr_max);
        if (index->tree.levels != levels || index->tree.nodes.size() != tree_node_count)
            throw Error("serialized tree shape does not match num_bins");
        index->degree_by_depth.assign(levels, index->M);
        if (version == 2)
        {
            for (std::uint32_t depth = 0; depth < levels; ++depth)
                read_scalar(input, index->degree_by_depth[depth], "degree_by_depth");
        }
        index->initialize_layout();

        for (std::uint32_t node_id = 0; node_id < tree_node_count; ++node_id)
        {
            std::array<std::int32_t, 7> serialized{};
            for (auto &value : serialized)
                read_scalar(input, value, "tree node metadata");
            const auto &node = index->tree.nodes[node_id];
            const std::array<std::int32_t, 7> expected = {
                node.node_id,
                node.leaf_l,
                node.leaf_r,
                node.depth,
                node.parent,
                node.left_child,
                node.right_child};
            if (serialized != expected)
                throw Error("serialized tree node metadata is inconsistent with a complete binary tree");
        }

        std::vector<std::uint32_t> graph_sizes(tree_node_count);
        for (auto &size : graph_sizes)
            read_scalar(input, size, "graph size");

        std::vector<double> serialized_attributes;
        read_array(input, serialized_attributes, index->num_points, "attributes");
        index->point_capacity = index->num_points;
        index->active_points = index->num_points;
        index->attributes.assign(index->point_capacity, 0);
        index->attribute_keys.assign(index->point_capacity, 0);
        for (PointId id = 0; id < index->num_points; ++id)
        {
            const double value = serialized_attributes[id];
            if (!std::isfinite(value) || std::trunc(value) != value ||
                value < std::numeric_limits<Attribute>::min() ||
                value > std::numeric_limits<Attribute>::max())
                throw Error("RFANNS requires integral int32 attributes");
            index->attributes[id] = static_cast<Attribute>(value);
            index->attribute_keys[id] = attribute_key(index->attributes[id]);
        }
        read_array(input, index->leaf_ids, index->point_capacity, "leaf ids");
        read_array(input,
                   index->degree,
                   static_cast<std::size_t>(index->point_capacity) * levels,
                   "degree array");
        read_edges(input,
                   index->edge_records,
                   static_cast<std::size_t>(index->point_capacity) * index->capacity_per_point,
                   index->attribute_keys,
                   index->num_points);

        index->visited.assign(index->point_capacity, 0);
        index->tree.reconstruct_members(index->leaf_ids,
                                        index->internal_to_external,
                                        index->num_points);
        for (std::uint32_t node_id = 0; node_id < tree_node_count; ++node_id)
        {
            if (graph_sizes[node_id] != index->tree.nodes[node_id].members.size())
                throw Error("serialized graph size does not match segment membership");
        }
        index->initialize_distance();
        return index;
    }

    Index Index::load(const std::string &index_path,
                      std::shared_ptr<const DenseDataset> dataset,
                      const std::string &id_map_path,
                      bool allow_data_superset,
                      std::uint64_t seed)
    {
        auto impl = IndexImpl::load(index_path,
                                    std::move(dataset),
                                    seed,
                                    allow_data_superset || !id_map_path.empty());
        if (!id_map_path.empty())
        {
            impl->internal_to_external = load_id_map(id_map_path, impl->num_points);
            impl->internal_to_external.resize(impl->point_capacity, kInvalidPointId);
            impl->rebuild_external_map();
            for (PointId id = 0; id < impl->num_points; ++id)
            {
                if (impl->internal_to_external[id] >= impl->dataset->size())
                    throw Error("ID map references a vector outside the supplied dataset");
            }
        }
        return Index(std::move(impl));
    }

    void IndexImpl::save(const std::string &path) const
    {
        if (active_points != num_points)
            throw Error("RFANNS v1/v2 cannot serialize deleted points; save is supported only when all points are active");
        if (attributes.size() < num_points || leaf_ids.size() < num_points)
            throw Error("point metadata is incomplete while saving index");
        if (degree.size() < static_cast<std::size_t>(num_points) * tree.levels ||
            edge_records.size() < static_cast<std::size_t>(num_points) * capacity_per_point)
            throw Error("graph metadata is incomplete while saving index");

        ensure_parent_directory(path);
        std::ofstream output(path, std::ios::binary);
        if (!output)
            throw Error("cannot open index for writing " + path);
        const std::uint32_t version = uniform_degree(*this) ? 1U : 2U;
        const std::uint32_t tree_node_count = static_cast<std::uint32_t>(tree.nodes.size());
        output.write(kIndexFileMagic.data(),
                     static_cast<std::streamsize>(kIndexFileMagic.size()));
        write_scalar(output, version);
        write_scalar(output, num_points);
        write_scalar(output, dimension);
        write_scalar(output, tree.num_bins);
        write_scalar(output, tree.levels);
        write_scalar(output, M);
        write_scalar(output, ef_construction);
        write_scalar(output, tree.attr_min);
        write_scalar(output, tree.attr_max);
        write_scalar(output, tree_node_count);
        if (version == 2)
        {
            for (const std::uint32_t value : degree_by_depth)
                write_scalar(output, value);
        }

        for (const auto &node : tree.nodes)
        {
            write_scalar(output, node.node_id);
            write_scalar(output, node.leaf_l);
            write_scalar(output, node.leaf_r);
            write_scalar(output, node.depth);
            write_scalar(output, node.parent);
            write_scalar(output, node.left_child);
            write_scalar(output, node.right_child);
        }
        for (const auto &node : tree.nodes)
        {
            const std::uint32_t graph_size =
                static_cast<std::uint32_t>(node.members.size());
            write_scalar(output, graph_size);
        }

        std::vector<double> serialized_attributes(num_points);
        for (PointId id = 0; id < num_points; ++id)
            serialized_attributes[id] = static_cast<double>(attributes[id]);
        output.write(reinterpret_cast<const char *>(serialized_attributes.data()),
                     static_cast<std::streamsize>(serialized_attributes.size() * sizeof(double)));
        output.write(reinterpret_cast<const char *>(leaf_ids.data()),
                     static_cast<std::streamsize>(num_points * sizeof(std::uint16_t)));
        output.write(reinterpret_cast<const char *>(degree.data()),
                     static_cast<std::streamsize>(static_cast<std::size_t>(num_points) * tree.levels * sizeof(std::uint16_t)));
        write_edge_destinations(output,
                                edge_records,
                                static_cast<std::size_t>(num_points) * capacity_per_point);
        if (!output)
            throw Error("failed while writing index " + path);
    }
}
