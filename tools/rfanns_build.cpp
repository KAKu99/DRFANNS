#include "cli.h"
#include "rfanns/index.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

namespace
{
    struct Workload
    {
        std::uint32_t operation = 0;
        std::uint32_t count = 0;
        std::uint32_t initial = 0;
        std::uint32_t final = 0;
        std::uint32_t dimension = 0;
        std::uint64_t seed = 0;
        std::vector<std::uint32_t> ids;
    };

    Workload load_workload(const std::string &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw rfanns::Error("cannot open active workload " + path);
        char magic[8]{};
        input.read(magic, sizeof(magic));
        const char expected[8] = {'R', 'F', 'D', 'Y', 'N', 'W', '1', '\0'};
        if (std::memcmp(magic, expected, sizeof(magic)) != 0)
            throw rfanns::Error("active workload magic is not RFDYNW1");
        std::uint32_t header[7]{};
        input.read(reinterpret_cast<char *>(header), sizeof(header));
        Workload workload;
        if (!input || header[0] != 1)
            throw rfanns::Error("unsupported active workload version");
        workload.operation = header[1];
        workload.count = header[2];
        workload.initial = header[3];
        workload.final = header[4];
        workload.dimension = header[5];
        workload.ids.resize(header[6]);
        input.read(reinterpret_cast<char *>(&workload.seed), sizeof(workload.seed));
        input.read(reinterpret_cast<char *>(workload.ids.data()),
                   static_cast<std::streamsize>(workload.ids.size() * sizeof(std::uint32_t)));
        if (!input)
            throw rfanns::Error("active workload is truncated");
        return workload;
    }
}

int main(int argc, char **argv)
{
    try
    {
        const rfanns::cli::Arguments args(argc, argv);
        args.allow_only({"data_path",
                         "attr_path",
                         "index_file",
                         "id_map_file",
                         "attr_min",
                         "attr_max",
                         "num_bins",
                         "min_segment_width",
                         "M",
                         "ef_construction",
                         "threads",
                         "active_workload",
                         "seed",
                         "validate"});

        const std::uint32_t requested_threads = args.u32(
            "threads",
            std::max<std::uint32_t>(1, std::thread::hardware_concurrency()));

        const rfanns::DenseDataset full_dataset =
            rfanns::DenseDataset::load(args.require("data_path"));
        const auto full_attributes =
            rfanns::load_attributes(args.require("attr_path"), full_dataset.size());
        std::shared_ptr<rfanns::DenseDataset> dataset;
        std::vector<rfanns::Attribute> attributes;
        std::vector<rfanns::ExternalId> active_ids;
        std::uint64_t workload_seed = 0;
        if (args.has("active_workload"))
        {
            const Workload workload = load_workload(args.get("active_workload"));
            if (workload.operation != 1 || workload.count != full_dataset.size() ||
                workload.dimension != full_dataset.dimension() ||
                workload.final != workload.count ||
                workload.initial + workload.ids.size() != workload.final)
                throw rfanns::Error("active workload does not match the full dataset");
            std::vector<unsigned char> excluded(workload.count, 0);
            for (const auto id : workload.ids)
            {
                if (id >= workload.count || excluded[id])
                    throw rfanns::Error("active workload contains an invalid or duplicate ID");
                excluded[id] = 1;
            }
            std::vector<float> values;
            values.reserve(static_cast<std::size_t>(workload.initial) * workload.dimension);
            attributes.reserve(workload.initial);
            active_ids.reserve(workload.initial);
            for (std::uint32_t id = 0; id < workload.count; ++id)
            {
                if (excluded[id])
                    continue;
                const float *row = full_dataset.row(id);
                values.insert(values.end(), row, row + workload.dimension);
                attributes.push_back(full_attributes[id]);
                active_ids.push_back(id);
            }
            dataset = std::make_shared<rfanns::DenseDataset>(
                workload.initial, workload.dimension, std::move(values));
            workload_seed = workload.seed;
        }
        else
        {
            dataset = std::make_shared<rfanns::DenseDataset>(full_dataset);
            attributes = full_attributes;
            active_ids.resize(full_dataset.size());
            std::iota(active_ids.begin(), active_ids.end(), 0);
        }
        rfanns::BuildOptions options;
        options.M = args.u32("M", 32);
        options.ef_construction = args.u32("ef_construction", 200);
        options.threads = requested_threads;
        options.seed = args.u64("seed", 0);

        if (args.has("min_segment_width"))
        {
            if (args.has("attr_min") || args.has("attr_max") || args.has("num_bins"))
                throw rfanns::Error("--min_segment_width cannot be combined with fixed-domain options");
            options.domain.mode = rfanns::DomainMode::Derived;
            options.domain.min_segment_width = args.real("min_segment_width", 0.0);
        }
        else
        {
            if (!args.has("attr_min") || !args.has("attr_max"))
                throw rfanns::Error("fixed domain requires --attr_min and --attr_max");
            options.domain.mode = rfanns::DomainMode::Fixed;
            options.domain.attr_min = args.real("attr_min", 0.0);
            options.domain.attr_max = args.real("attr_max", 0.0);
            options.domain.num_bins = args.u32("num_bins", 2048);
        }

        const auto started = std::chrono::steady_clock::now();
        rfanns::Index index = rfanns::IndexBuilder::build(dataset, attributes, options);
        const auto built = std::chrono::steady_clock::now();
        if (args.boolean("validate", false))
            index.validate();
        const auto validated = std::chrono::steady_clock::now();
        // The builder may compact a non-prefix active set, so the sidecar must
        // be written from active_ids rather than from Index's temporary
        // identity mapping.
        index.save(args.require("index_file"));
        if (args.has("id_map_file"))
            rfanns::save_id_map(args.get("id_map_file"), active_ids);
        const auto saved = std::chrono::steady_clock::now();
        const double build_seconds =
            std::chrono::duration<double>(built - started).count();
        const double validation_seconds =
            std::chrono::duration<double>(validated - built).count();
        const double save_seconds =
            std::chrono::duration<double>(saved - validated).count();
        // The reported construction metric includes graph construction,
        // eager reverse-state construction, and serialization, but excludes
        // optional validation. Validation is reported separately.
        const double construction_seconds = build_seconds + save_seconds;
        const double total_seconds =
            std::chrono::duration<double>(saved - started).count();
        const auto metadata = index.metadata();
        const auto &update_stats = index.update_stats();
        std::cout << "construction_seconds:" << construction_seconds << '\n'
                  << "index_build_seconds:" << build_seconds << '\n'
                  << "validation_seconds:" << validation_seconds << '\n'
                  << "save_seconds:" << save_seconds << '\n'
                  << "total_seconds_including_validation:" << total_seconds << '\n'
                  << "num_points:" << metadata.num_points << '\n'
                  << "active_count:" << active_ids.size() << '\n'
                  << "id_space_points:" << full_dataset.size() << '\n'
                  << "active_workload:" << args.get("active_workload") << '\n'
                  << "workload_seed:" << workload_seed << '\n'
                  << "dimension:" << metadata.dimension << '\n'
                  << "num_bins:" << metadata.num_bins << '\n'
                  << "levels:" << metadata.levels << '\n'
                  << "attr_min:" << metadata.attr_min << '\n'
                  << "attr_max:" << metadata.attr_max << '\n'
                  << "M:" << metadata.M << '\n'
                  << "ef_construction:" << metadata.ef_construction << '\n'
                  << "threads:" << options.threads << '\n'
                  << "construction_parallelism:point_parallel\n"
                  << "reverse_adjacency_edges:" << update_stats.reverse_adjacency_edges << '\n'
                  << "reverse_adjacency_build_seconds:"
                  << update_stats.reverse_adjacency_build_seconds << '\n'
                  << "tree_partition:uniform_attribute_bins\n"
                  << "insertion_route:leaf_to_root\n"
                  << "neighbor_selection:hnsw_heuristic2\n"
                  << "reverse_state:eager_exact\n"
                  << "search_route:attribute_aware_edges\n"
                  << "index_bytes:" << std::filesystem::file_size(args.require("index_file")) << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
