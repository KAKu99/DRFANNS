#include "cli.h"
#include "rfanns/dataset.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace
{
    struct Workload
    {
        std::uint32_t operation{0};
        std::uint32_t id_space_points{0};
        std::uint32_t initial_active{0};
        std::uint32_t final_active{0};
        std::uint32_t dimension{0};
        std::uint64_t seed{0};
        std::vector<rfanns::ExternalId> ids;
    };

    template <typename T>
    void read_scalar(std::ifstream &input, T &value, const std::string &name)
    {
        input.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!input)
            throw rfanns::Error("workload ended while reading " + name);
    }

    Workload load_workload(const std::string &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw rfanns::Error("cannot open workload " + path);
        char magic[8]{};
        input.read(magic, sizeof(magic));
        const char expected[8] = {'R', 'F', 'D', 'Y', 'N', 'W', '1', '\0'};
        if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0)
            throw rfanns::Error("workload magic is not RFDYNW1");

        std::uint32_t version = 0;
        std::uint32_t operation_count = 0;
        Workload result;
        read_scalar(input, version, "version");
        read_scalar(input, result.operation, "operation");
        read_scalar(input, result.id_space_points, "id_space_points");
        read_scalar(input, result.initial_active, "initial_active");
        read_scalar(input, result.final_active, "final_active");
        read_scalar(input, result.dimension, "dimension");
        read_scalar(input, operation_count, "operation_count");
        read_scalar(input, result.seed, "seed");
        if (version != 1 || (result.operation != 1 && result.operation != 2))
            throw rfanns::Error("unsupported workload version or operation");
        result.ids.resize(operation_count);
        input.read(reinterpret_cast<char *>(result.ids.data()),
                   static_cast<std::streamsize>(result.ids.size()
                                                * sizeof(rfanns::ExternalId)));
        if (!input)
            throw rfanns::Error("workload ended while reading operation IDs");
        return result;
    }

    float squared_l2(const float *left, const float *right, std::uint32_t dimension)
    {
        float result = 0.0f;
#pragma omp simd reduction(+ : result)
        for (std::uint32_t component = 0; component < dimension; ++component)
        {
            const float difference = left[component] - right[component];
            result += difference * difference;
        }
        return result;
    }
}

int main(int argc, char **argv)
{
    try
    {
        const rfanns::cli::Arguments args(argc, argv);
        args.allow_only({"data_path",
                         "query_path",
                         "range_path",
                         "attr_path",
                         "groundtruth_file",
                         "workload_file",
                         "manifest_file",
                         "operation",
                         "fraction",
                         "seed",
                         "K",
                         "Nq",
                         "threads"});
        const rfanns::DenseDataset data = rfanns::DenseDataset::load(args.require("data_path"));
        const rfanns::DenseDataset queries = rfanns::DenseDataset::load(args.require("query_path"));
        if (queries.dimension() != data.dimension())
            throw rfanns::Error("query and data dimensions differ");
        const std::uint32_t query_count = args.u32("Nq", queries.size());
        const std::uint32_t k = args.u32("K", 10);
        const std::uint32_t threads = args.u32("threads", 1);
        if (query_count == 0 || query_count > queries.size() || k == 0 || threads == 0)
            throw rfanns::Error("invalid groundtruth parameters");
        const auto ranges = rfanns::load_ranges(args.require("range_path"), query_count);
        const auto attributes = rfanns::load_attributes(args.require("attr_path"), data.size());
        std::vector<std::uint8_t> inactive(data.size(), 0);
        std::uint32_t active_count = static_cast<std::uint32_t>(data.size());
        std::uint64_t workload_seed = 0;
        std::uint32_t workload_operation = 0;
        if (args.has("workload_file"))
        {
            const Workload workload = load_workload(args.require("workload_file"));
            if (workload.id_space_points != data.size()
                || workload.dimension != data.dimension())
                throw rfanns::Error("workload does not match the dataset");
            const bool counts_valid =
                (workload.operation == 1
                 && workload.initial_active + workload.ids.size() == workload.final_active
                 && workload.final_active == data.size())
                || (workload.operation == 2
                    && workload.final_active + workload.ids.size() == workload.initial_active
                    && workload.initial_active == data.size());
            if (!counts_valid)
                throw rfanns::Error("workload active counts are inconsistent");
            if (args.has("operation"))
            {
                const std::string requested = args.get("operation");
                const std::string actual = workload.operation == 1 ? "insert" : "delete";
                if (requested != actual)
                    throw rfanns::Error("--operation does not match the workload");
            }
            if (args.has("seed") && args.u64("seed", 0) != workload.seed)
                throw rfanns::Error("--seed does not match the workload");
            if (args.has("fraction"))
            {
                const auto expected_operations = static_cast<std::size_t>(std::llround(
                    args.real("fraction", 0.0) * static_cast<double>(data.size())));
                if (expected_operations != workload.ids.size())
                    throw rfanns::Error("--fraction does not match the workload size");
            }
            for (const auto id : workload.ids)
            {
                if (id >= data.size() || inactive[id])
                    throw rfanns::Error("workload contains an invalid or duplicate ID");
                inactive[id] = 1;
            }
            active_count = workload.operation == 1
                               ? workload.initial_active
                               : workload.final_active;
            workload_seed = workload.seed;
            workload_operation = workload.operation;
        }
        std::vector<std::vector<rfanns::ExternalId>> result(query_count,
                                                            std::vector<rfanns::ExternalId>(k,
                                                                                           rfanns::kInvalidPointId));

#pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (std::int32_t query_id = 0;
             query_id < static_cast<std::int32_t>(query_count);
             ++query_id)
        {
            using Candidate = std::pair<float, rfanns::ExternalId>;
            std::priority_queue<Candidate> top;
            const auto range = ranges[static_cast<std::size_t>(query_id)];
            for (rfanns::ExternalId id = 0; id < data.size(); ++id)
            {
                if (inactive[id])
                    continue;
                if (attributes[id] < range.low || attributes[id] > range.high)
                    continue;
                const float distance = squared_l2(queries.row(query_id),
                                                  data.row(id),
                                                  data.dimension());
                top.emplace(distance, id);
                if (top.size() > k)
                    top.pop();
            }
            std::vector<Candidate> ordered;
            ordered.reserve(top.size());
            while (!top.empty())
            {
                ordered.push_back(top.top());
                top.pop();
            }
            std::sort(ordered.begin(), ordered.end());
            for (std::size_t rank = 0; rank < ordered.size(); ++rank)
                result[query_id][rank] = ordered[rank].second;
        }

        rfanns::ensure_parent_directory(args.require("groundtruth_file"));
        std::ofstream output(args.require("groundtruth_file"), std::ios::binary);
        if (!output)
            throw rfanns::Error("cannot open groundtruth output");
        const std::int32_t count_value = static_cast<std::int32_t>(query_count);
        const std::int32_t k_value = static_cast<std::int32_t>(k);
        output.write(reinterpret_cast<const char *>(&count_value), sizeof(count_value));
        output.write(reinterpret_cast<const char *>(&k_value), sizeof(k_value));
        for (const auto &row : result)
            output.write(reinterpret_cast<const char *>(row.data()),
                         static_cast<std::streamsize>(row.size() * sizeof(rfanns::ExternalId)));
        if (!output)
            throw rfanns::Error("failed while writing groundtruth output");
        output.close();

        if (args.has("manifest_file"))
        {
            rfanns::ensure_parent_directory(args.get("manifest_file"));
            std::ofstream manifest(args.get("manifest_file"));
            if (!manifest)
                throw rfanns::Error("cannot open groundtruth manifest output");
            manifest << "operation,fraction,seed,id_space_points,inactive_count,active_count,"
                        "query_count,k,threads,distance,groundtruth_file,workload_file\n"
                     << (workload_operation == 1 ? "insert"
                                                 : workload_operation == 2 ? "delete" : "none")
                     << ',' << args.get("fraction") << ',' << workload_seed << ',' << data.size()
                     << ',' << (data.size() - active_count) << ',' << active_count << ','
                     << query_count << ',' << k << ',' << threads << ",squared_l2,"
                     << args.require("groundtruth_file") << ',' << args.get("workload_file") << '\n';
            if (!manifest)
                throw rfanns::Error("failed while writing groundtruth manifest");
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
