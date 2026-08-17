#include "cli.h"
#include "rfanns/index.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv)
{
    try
    {
        const rfanns::cli::Arguments args(argc, argv);
        args.allow_only({"data_path",
                         "query_path",
                         "range_path",
                         "groundtruth_path",
                         "index_file",
                         "id_map_file",
                         "result_file",
                         "K",
                         "Nq",
                         "ef",
                         "seed"});
        const auto data = std::make_shared<rfanns::DenseDataset>(
            rfanns::DenseDataset::load(args.require("data_path")));
        const rfanns::DenseDataset queries =
            rfanns::DenseDataset::load(args.require("query_path"));
        if (queries.dimension() != data->dimension())
            throw rfanns::Error("query dimension does not match data dimension");
        const std::uint32_t query_count = args.u32("Nq", queries.size());
        if (query_count == 0 || query_count > queries.size())
            throw rfanns::Error("Nq is outside the loaded query count");
        const std::uint32_t k = args.u32("K", 10);
        const auto ranges = rfanns::load_ranges(args.require("range_path"), query_count);
        const auto groundtruth = rfanns::load_groundtruth(args.require("groundtruth_path"),
                                                          query_count,
                                                          k);
        const auto ef_values = rfanns::cli::parse_u32_list(args.require("ef"), "ef");
        rfanns::Index index = rfanns::Index::load(args.require("index_file"),
                                                  data,
                                                  args.get("id_map_file"),
                                                  false,
                                                  args.u64("seed", 0));
        index.prepare_search();

        rfanns::ensure_parent_directory(args.require("result_file"));
        std::ofstream output(args.require("result_file"));
        if (!output)
            throw rfanns::Error("cannot open result file");
        output << "ef,recall,qps,queries,search_seconds\n" << std::setprecision(12);
        for (const std::uint32_t ef : ef_values)
        {
            std::uint64_t true_positives = 0;
            double search_seconds = 0.0;
            rfanns::SearchOptions options;
            options.k = k;
            options.ef = ef;
            for (std::uint32_t query_id = 0; query_id < query_count; ++query_id)
            {
                const auto started = std::chrono::steady_clock::now();
                const auto result = index.search(queries.row(query_id),
                                                 query_id,
                                                 ranges[query_id],
                                                 options);
                search_seconds += std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - started)
                                      .count();
                for (const auto &item : result)
                {
                    if (std::find(groundtruth[query_id].begin(),
                                  groundtruth[query_id].end(),
                                  item.id) != groundtruth[query_id].end())
                        ++true_positives;
                }
            }
            const double recall = static_cast<double>(true_positives) /
                                  (static_cast<double>(query_count) * k);
            const double qps = query_count / search_seconds;
            output << ef << ',' << recall << ',' << qps << ',' << query_count << ','
                   << search_seconds << '\n';
            std::cout << "ef=" << ef << " recall=" << recall << " qps=" << qps << '\n';
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
