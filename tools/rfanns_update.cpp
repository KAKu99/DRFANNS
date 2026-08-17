#include "cli.h"
#include "rfanns/index.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>

namespace
{
    enum class Operation : std::uint32_t
    {
        Insert = 1,
        Delete = 2
    };

    struct Workload
    {
        Operation operation{Operation::Insert};
        std::uint32_t id_space_points{0};
        std::uint32_t initial_active{0};
        std::uint32_t final_active{0};
        std::uint32_t dimension{0};
        std::uint64_t seed{0};
        std::vector<rfanns::ExternalId> ids;
    };

    struct EvaluationSet
    {
        std::string tag;
        std::string range_path;
        std::string groundtruth_path;
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
        char magic[8] = {};
        input.read(magic, sizeof(magic));
        const char expected[8] = {'R', 'F', 'D', 'Y', 'N', 'W', '1', '\0'};
        if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0)
            throw rfanns::Error("workload magic is not RFDYNW1");
        std::uint32_t version = 0;
        std::uint32_t operation = 0;
        std::uint32_t operation_count = 0;
        Workload result;
        read_scalar(input, version, "version");
        read_scalar(input, operation, "operation");
        read_scalar(input, result.id_space_points, "id_space_points");
        read_scalar(input, result.initial_active, "initial_active");
        read_scalar(input, result.final_active, "final_active");
        read_scalar(input, result.dimension, "dimension");
        read_scalar(input, operation_count, "operation_count");
        read_scalar(input, result.seed, "seed");
        if (version != 1 || (operation != 1 && operation != 2))
            throw rfanns::Error("unsupported workload version or operation");
        result.operation = static_cast<Operation>(operation);
        result.ids.resize(operation_count);
        input.read(reinterpret_cast<char *>(result.ids.data()),
                   static_cast<std::streamsize>(result.ids.size() * sizeof(rfanns::ExternalId)));
        if (!input)
            throw rfanns::Error("workload ended while reading operation IDs");
        return result;
    }

    std::vector<EvaluationSet> load_evaluation_manifest(const std::string &path)
    {
        std::ifstream input(path);
        if (!input)
            throw rfanns::Error("cannot open evaluation manifest " + path);
        std::vector<EvaluationSet> result;
        std::string line;
        while (std::getline(input, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            std::stringstream stream(line);
            EvaluationSet item;
            if (!std::getline(stream, item.tag, '\t') ||
                !std::getline(stream, item.range_path, '\t') ||
                !std::getline(stream, item.groundtruth_path, '\t'))
                throw rfanns::Error("evaluation manifest must be tab-separated: tag, range_path, groundtruth_path");
            if (item.tag == "tag" && item.range_path == "range_path")
                continue;
            if (item.tag.empty() || item.range_path.empty() || item.groundtruth_path.empty())
                throw rfanns::Error("evaluation manifest contains an empty field");
            result.push_back(std::move(item));
        }
        if (result.empty())
            throw rfanns::Error("evaluation manifest contains no evaluation sets");
        return result;
    }

    double percentile(std::vector<double> values, double fraction)
    {
        if (values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        const double position = fraction * (values.size() - 1);
        const std::size_t low = static_cast<std::size_t>(std::floor(position));
        const std::size_t high = static_cast<std::size_t>(std::ceil(position));
        const double weight = position - low;
        return values[low] * (1.0 - weight) + values[high] * weight;
    }
}

int main(int argc, char **argv)
{
    try
    {
        const rfanns::cli::Arguments args(argc, argv);
        args.allow_only({"data_path",
                         "attr_path",
                         "query_path",
                         "index_file",
                         "id_map_file",
                         "workload_file",
                         "evaluation_manifest",
                         "result_csv",
                         "summary_csv",
                         "per_operation_csv",
                         "save_index_file",
                         "save_id_map_file",
                         "dynamic_attr_expand",
                         "K",
                         "Nq",
                         "ef",
                         "seed",
                         "skip_evaluation"});
        if (args.has("save_index_file") != args.has("save_id_map_file"))
            throw rfanns::Error("--save_index_file and --save_id_map_file must be supplied together");

        const auto data = std::make_shared<rfanns::DenseDataset>(
            rfanns::DenseDataset::load(args.require("data_path")));
        const auto attributes = rfanns::load_attributes(args.require("attr_path"), data->size());
        const bool skip_evaluation = args.boolean("skip_evaluation", false);
        std::unique_ptr<rfanns::DenseDataset> queries;
        std::uint32_t query_count = 0;
        if (!skip_evaluation)
        {
            queries = std::make_unique<rfanns::DenseDataset>(
                rfanns::DenseDataset::load(args.require("query_path")));
            query_count = args.u32("Nq", queries->size());
            if (query_count == 0 || query_count > queries->size() ||
                queries->dimension() != data->dimension())
                throw rfanns::Error("invalid query count or dimension");
        }
        const std::uint32_t k = args.u32("K", 10);
        const auto ef_values = skip_evaluation
                                   ? std::vector<std::uint32_t>{}
                                   : rfanns::cli::parse_u32_list(args.require("ef"), "ef");
        const Workload workload = load_workload(args.require("workload_file"));
        if (workload.dimension != data->dimension() || workload.id_space_points > data->size())
            throw rfanns::Error("workload does not match the supplied dataset");
        const bool workload_counts_valid =
            (workload.operation == Operation::Insert &&
             workload.initial_active + workload.ids.size() == workload.final_active &&
             workload.final_active == workload.id_space_points) ||
            (workload.operation == Operation::Delete &&
             workload.final_active + workload.ids.size() == workload.initial_active &&
             workload.initial_active == workload.id_space_points);
        if (!workload_counts_valid)
            throw rfanns::Error("workload active counts are inconsistent");
        std::vector<rfanns::ExternalId> sorted_workload_ids = workload.ids;
        std::sort(sorted_workload_ids.begin(), sorted_workload_ids.end());
        if ((!sorted_workload_ids.empty() &&
             sorted_workload_ids.back() >= workload.id_space_points) ||
            std::adjacent_find(sorted_workload_ids.begin(), sorted_workload_ids.end()) !=
                sorted_workload_ids.end())
            throw rfanns::Error("workload contains an out-of-range or duplicate ID");
        const auto evaluation_sets = skip_evaluation
                                         ? std::vector<EvaluationSet>{}
                                         : load_evaluation_manifest(args.require("evaluation_manifest"));

        rfanns::Index index = rfanns::Index::load(args.require("index_file"),
                                                  data,
                                                  args.get("id_map_file"),
                                                  true,
                                                  args.u64("seed", 0));
        const rfanns::IndexMetadata initial = index.metadata();
        if (initial.active_points != workload.initial_active)
            throw rfanns::Error("workload initial_active does not match the loaded checkpoint");
        index.reserve(std::max<std::uint32_t>(workload.id_space_points,
                                              initial.num_points + workload.ids.size()));
        const auto update_preparation_started = std::chrono::steady_clock::now();
        // Materialize all update-only state before the operation timer.  In
        // particular, a loaded v1/v2 index does not serialize reverse
        // adjacency, so rebuilding it is preparation rather than insert/delete
        // latency.
        index.prepare_updates();
        const double update_preparation_seconds = std::chrono::duration<double>(
                                                      std::chrono::steady_clock::now() -
                                                      update_preparation_started)
                                                      .count();

        const bool dynamic_expand = args.boolean("dynamic_attr_expand", false);
        std::vector<double> operation_seconds;
        operation_seconds.reserve(workload.ids.size());
        const auto operation_started = std::chrono::steady_clock::now();
        for (const rfanns::ExternalId external_id : workload.ids)
        {
            if (external_id >= data->size())
                throw rfanns::Error("workload external ID is outside the dataset");
            const auto started = std::chrono::steady_clock::now();
            if (workload.operation == Operation::Insert)
            {
                index.insert(external_id,
                             data->row(external_id),
                             attributes[external_id],
                             rfanns::InsertOptions{dynamic_expand});
            }
            else if (!index.erase(external_id))
            {
                throw rfanns::Error("delete external ID is absent or already inactive");
            }
            operation_seconds.push_back(std::chrono::duration<double>(
                                                   std::chrono::steady_clock::now() - started)
                                                   .count());
        }
        const double operation_total_seconds = std::chrono::duration<double>(
                                                   std::chrono::steady_clock::now() - operation_started)
                                                   .count();
        index.validate();

        if (!skip_evaluation)
        {
            rfanns::ensure_parent_directory(args.require("result_csv"));
            std::ofstream results(args.require("result_csv"));
            if (!results)
                throw rfanns::Error("cannot open result CSV");
            results << "selectivity,phase,ef,queries,search_seconds,qps,total_results,recall\n"
                    << std::setprecision(12);
            for (const auto &evaluation : evaluation_sets)
            {
                const auto ranges = rfanns::load_ranges(evaluation.range_path, query_count);
                const auto groundtruth = rfanns::load_groundtruth(evaluation.groundtruth_path,
                                                                  query_count,
                                                                  k);
                for (const std::uint32_t ef : ef_values)
                {
                    rfanns::SearchOptions search_options{k, ef};
                    std::uint64_t true_positives = 0;
                    std::uint64_t total_results = 0;
                    double seconds = 0.0;
                    for (std::uint32_t query_id = 0; query_id < query_count; ++query_id)
                    {
                        const auto started = std::chrono::steady_clock::now();
                        const auto found = index.search(queries->row(query_id),
                                                        query_id,
                                                        ranges[query_id],
                                                        search_options);
                        seconds += std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
                        total_results += found.size();
                        for (const auto &item : found)
                        {
                            if (std::find(groundtruth[query_id].begin(),
                                          groundtruth[query_id].end(),
                                          item.id) != groundtruth[query_id].end())
                                ++true_positives;
                        }
                    }
                    const double recall = static_cast<double>(true_positives) /
                                          (static_cast<double>(query_count) * k);
                    results << evaluation.tag << ",post_"
                            << (workload.operation == Operation::Insert ? "insert" : "delete")
                            << "," << ef << ',' << query_count << ',' << seconds << ','
                            << query_count / seconds << ',' << total_results << ',' << recall << '\n';
                }
            }
        }

        if (args.has("per_operation_csv"))
        {
            rfanns::ensure_parent_directory(args.get("per_operation_csv"));
            std::ofstream per_operation(args.get("per_operation_csv"));
            per_operation << "ordinal,operation,external_id,operation_seconds,operation_ms\n"
                          << std::setprecision(12);
            for (std::size_t index_value = 0; index_value < operation_seconds.size(); ++index_value)
            {
                per_operation << index_value << ','
                              << (workload.operation == Operation::Insert ? "insert" : "delete") << ','
                              << workload.ids[index_value] << ',' << operation_seconds[index_value] << ','
                              << operation_seconds[index_value] * 1000.0 << '\n';
            }
        }

        if (args.has("save_index_file"))
            index.save(args.get("save_index_file"), args.get("save_id_map_file"));

        const auto final_metadata = index.metadata();
        const auto &stats = index.update_stats();
        const double operation_sum_seconds = std::accumulate(
            operation_seconds.begin(), operation_seconds.end(), 0.0);
        const double average_ms = operation_seconds.empty()
                                      ? 0.0
                                      : operation_sum_seconds * 1000.0 / operation_seconds.size();
        const auto minmax_seconds = std::minmax_element(
            operation_seconds.begin(), operation_seconds.end());
        const double minimum_ms = operation_seconds.empty() ? 0.0 : *minmax_seconds.first * 1000.0;
        const double maximum_ms = operation_seconds.empty() ? 0.0 : *minmax_seconds.second * 1000.0;
        rfanns::ensure_parent_directory(args.require("summary_csv"));
        std::ofstream summary(args.require("summary_csv"));
        if (!summary)
            throw rfanns::Error("cannot open summary CSV");
        summary << "metric,value\n" << std::setprecision(12)
                << "operation," << (workload.operation == Operation::Insert ? "insert" : "delete") << '\n'
                << "operation_completed," << operation_seconds.size() << '\n'
                << "update_preparation_seconds," << update_preparation_seconds << '\n'
                << "operation_total_seconds," << operation_total_seconds << '\n'
                << "operation_avg_ms," << average_ms << '\n'
                << "operation_p50_ms," << percentile(operation_seconds, 0.50) * 1000.0 << '\n'
                << "operation_p95_ms," << percentile(operation_seconds, 0.95) * 1000.0 << '\n'
                << "operation_p99_ms," << percentile(operation_seconds, 0.99) * 1000.0 << '\n'
                << "operation_min_ms," << minimum_ms << '\n'
                << "operation_max_ms," << maximum_ms << '\n'
                << "operation_ops_per_second,"
                << (operation_total_seconds > 0.0
                        ? operation_seconds.size() / operation_total_seconds
                        : 0.0) << '\n'
                << "initial_points," << initial.num_points << '\n'
                << "initial_active," << initial.active_points << '\n'
                << "final_points," << final_metadata.num_points << '\n'
                << "final_active," << final_metadata.active_points << '\n'
                << "dynamic_attr_expand," << (dynamic_expand ? 1 : 0) << '\n'
                << "attr_expand_count," << stats.attr_expand_count << '\n'
                << "attr_expand_left_count," << stats.attr_expand_left_count << '\n'
                << "attr_expand_right_count," << stats.attr_expand_right_count << '\n'
                << "attr_expand_seconds," << stats.attr_expand_seconds << '\n'
                << "final_attr_min," << final_metadata.attr_min << '\n'
                << "final_attr_max," << final_metadata.attr_max << '\n'
                << "final_num_bins," << final_metadata.num_bins << '\n'
                << "final_levels," << final_metadata.levels << '\n'
                << "reverse_adjacency_edges," << stats.reverse_adjacency_edges << '\n'
                << "reverse_adjacency_build_seconds," << stats.reverse_adjacency_build_seconds << '\n'
                << "reverse_adjacency_update_seconds," << stats.reverse_adjacency_update_seconds << '\n'
                << "reverse_adjacency_updates," << stats.reverse_adjacency_updates << '\n'
                << "forward_lists_rewritten," << stats.forward_lists_rewritten << '\n'
                << "forward_list_rewrite_seconds," << stats.forward_list_rewrite_seconds << '\n'
                << "delete_segments_processed," << stats.delete_segments_processed << '\n'
                << "delete_search_candidates," << stats.delete_search_candidates << '\n'
                << "delete_incoming_owners," << stats.delete_incoming_owners << '\n'
                << "delete_outgoing_edges_removed," << stats.delete_outgoing_edges_removed << '\n'
                << "delete_incoming_edges_removed," << stats.delete_incoming_edges_removed << '\n'
                << "delete_replacement_edges_added," << stats.delete_replacement_edges_added << '\n'
                << "delete_edge_lists_rewritten," << stats.delete_edge_lists_rewritten << '\n'
                << "delete_remaining_edges," << stats.delete_remaining_edges << '\n'
                << "save_index_file," << args.get("save_index_file") << '\n'
                << "save_id_map_file," << args.get("save_id_map_file") << '\n';

        std::cout << "operation_completed:" << operation_seconds.size() << '\n'
                  << "update_preparation_seconds:" << update_preparation_seconds << '\n'
                  << "operation_total_seconds:" << operation_total_seconds << '\n'
                  << "final_active:" << final_metadata.active_points << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
