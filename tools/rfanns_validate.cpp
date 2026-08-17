#include "cli.h"
#include "rfanns/index.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <sys/resource.h>

namespace
{
    long peak_rss_kib()
    {
        rusage usage{};
        return getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : -1;
    }
}

int main(int argc, char **argv)
{
    try
    {
        const rfanns::cli::Arguments args(argc, argv);
        args.allow_only({"data_path", "index_file", "id_map_file", "seed"});

        const auto started = std::chrono::steady_clock::now();
        const auto data = std::make_shared<rfanns::DenseDataset>(
            rfanns::DenseDataset::load(args.require("data_path")));
        rfanns::Index index = rfanns::Index::load(args.require("index_file"),
                                                  data,
                                                  args.get("id_map_file"),
                                                  false,
                                                  args.u64("seed", 0));
        // Load files do not contain reverse adjacency. Build update-ready
        // state so validation covers both directions of every edge.
        index.prepare_updates();
        index.validate();
        const auto metadata = index.metadata();
        const auto &update_stats = index.update_stats();
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();

        std::cout << "validation:passed\n"
                  << "points:" << metadata.num_points << '\n'
                  << "active_points:" << metadata.active_points << '\n'
                  << "dimension:" << metadata.dimension << '\n'
                  << "M:" << metadata.M << '\n'
                  << "ef_construction:" << metadata.ef_construction << '\n'
                  << "num_bins:" << metadata.num_bins << '\n'
                  << "levels:" << metadata.levels << '\n'
                  << "attr_min:" << metadata.attr_min << '\n'
                  << "attr_max:" << metadata.attr_max << '\n'
                  << "reverse_adjacency_edges:" << update_stats.reverse_adjacency_edges << '\n'
                  << "reverse_adjacency_build_seconds:"
                  << update_stats.reverse_adjacency_build_seconds << '\n'
                  << "seconds:" << seconds << '\n'
                  << "max_rss_kib:" << peak_rss_kib() << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "validation:failed\nerror:" << error.what() << '\n';
        return 1;
    }
}
