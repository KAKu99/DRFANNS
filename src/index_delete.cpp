#include "index_internal.h"

#include <algorithm>
#include <limits>

namespace rfanns
{
    namespace
    {
        bool append_unique(std::vector<PointId> &values, PointId id)
        {
            if (std::find(values.begin(), values.end(), id) != values.end())
                return false;
            values.push_back(id);
            return true;
        }
    }

    std::vector<std::vector<PointId>> IndexImpl::collect_outgoing(PointId deleted) const
    {
        std::vector<std::vector<PointId>> result(tree.levels);
        for (std::uint32_t depth = 0; depth < tree.levels; ++depth)
        {
            auto &neighbors_at_depth = result[depth];
            const std::uint16_t count = get_degree(deleted, depth);
            neighbors_at_depth.reserve(count);
            for (std::uint16_t slot = 0; slot < count; ++slot)
                neighbors_at_depth.push_back(get_neighbor(deleted, depth, slot));
        }
        return result;
    }

    std::vector<PointId> IndexImpl::repair_search(PointId deleted,
                                                  std::uint32_t depth)
    {
        const std::uint32_t repair_ef = ef_construction == 0 ? 200 : ef_construction;
        const std::uint32_t owner_limit = repair_ef;
        std::vector<PointId> result;
        if (repair_ef == 0 || depth >= tree.levels)
            return result;

        const std::uint32_t tag = next_visited_tag();

        std::priority_queue<DistanceId, std::vector<DistanceId>, std::greater<DistanceId>> candidates;
        std::priority_queue<DistanceId> owners;
        auto push = [&](PointId id) {
            if (visited[id] == tag)
                return;
            visited[id] = tag;
            float value = 0.0f;
            if (id != deleted)
            {
                value = distance(deleted, id);
                owners.emplace(value, id);
                if (owners.size() > owner_limit)
                    owners.pop();
            }
            candidates.emplace(value, id);
        };

        push(deleted);
        std::uint32_t expansions = 0;
        while (!candidates.empty() && expansions < repair_ef)
        {
            const PointId current = candidates.top().second;
            candidates.pop();
            ++expansions;
            for (std::uint16_t slot = 0; slot < get_degree(current, depth); ++slot)
                push(get_neighbor(current, depth, slot));
        }
        result.reserve(owners.size());
        while (!owners.empty())
        {
            result.push_back(owners.top().second);
            owners.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    std::vector<PointId> IndexImpl::incoming_owners(PointId deleted,
                                                    std::uint32_t depth) const
    {
        if (!reverse_adjacency_ready)
            throw Error("delete requires prepared reverse adjacency");
        const auto &incoming = reverse_adjacency[edge_list_index(deleted, depth)];
        return {incoming.begin(), incoming.end()};
    }

    std::vector<PointId> IndexImpl::select_repaired_list(
        PointId owner,
        std::uint32_t depth,
        const std::vector<PointId> &survivors,
        const std::vector<PointId> &forced,
        const std::vector<PointId> &pool,
        std::uint32_t target_degree) const
    {
        const std::uint32_t capacity = capacity_at_depth(depth);
        target_degree = std::min<std::uint32_t>(capacity,
                                                std::max<std::uint32_t>(
                                                    target_degree,
                                                    static_cast<std::uint32_t>(survivors.size())));
        std::vector<PointId> selected;
        selected.reserve(capacity);
        auto add = [&](PointId candidate) {
            if (selected.size() >= capacity || candidate == owner)
                return false;
            return append_unique(selected, candidate);
        };

        std::vector<PointId> forced_unique;
        for (PointId candidate : forced)
        {
            if (candidate != owner)
                append_unique(forced_unique, candidate);
        }
        std::sort(forced_unique.begin(), forced_unique.end(), [&](PointId left, PointId right) {
            const float a = distance(owner, left);
            const float b = distance(owner, right);
            return a == b ? left < right : a < b;
        });
        for (PointId candidate : forced_unique)
            add(candidate);
        for (PointId candidate : survivors)
        {
            if (selected.size() >= target_degree)
                break;
            add(candidate);
        }

        std::vector<DistanceId> scored;
        scored.reserve(pool.size());
        for (PointId candidate : pool)
        {
            if (candidate == owner ||
                std::find(selected.begin(), selected.end(), candidate) != selected.end())
                continue;
            scored.emplace_back(distance(owner, candidate), candidate);
        }
        std::sort(scored.begin(), scored.end());
        for (const auto &candidate : scored)
        {
            if (selected.size() >= target_degree)
                break;
            add(candidate.second);
        }
        return selected;
    }

    void IndexImpl::repair_delete_depth(PointId deleted,
                                        std::uint32_t depth,
                                        const std::vector<PointId> &outgoing,
                                        const std::vector<PointId> &owners,
                                        const std::vector<PointId> &search_candidates,
                                        DeleteLocalStats &local)
    {
        auto record_rewrite = [&](const NeighborRewriteSummary &summary) {
            local.reverse_edge_count_change += summary.reverse_edge_count_change;
            local.reverse_updates += summary.reverse_updates;
            local.forward_lists_rewritten += summary.forward_lists_rewritten;
        };
        local.outgoing_removed += get_degree(deleted, depth);
        record_rewrite(write_neighbor_list(deleted, depth, {}));

        std::vector<PointId> recipients;
        recipients.reserve(owners.size() + search_candidates.size());
        for (PointId owner : owners)
            recipients.push_back(owner);
        const std::vector<PointId> &outgoing_unique = outgoing;
        for (PointId candidate : search_candidates)
        {
            if (recipients.size() >= outgoing_unique.size())
                break;
            if (candidate != deleted)
                append_unique(recipients, candidate);
        }

        std::vector<std::vector<PointId>> forced(recipients.size());
        for (PointId target : outgoing_unique)
        {
            bool already_matched = false;
            for (PointId owner : recipients)
            {
                if (owner != target)
                {
                    bool has_edge = false;
                    for (std::uint16_t slot = 0; slot < get_degree(owner, depth); ++slot)
                    {
                        if (get_neighbor(owner, depth, slot) == target)
                        {
                            has_edge = true;
                            break;
                        }
                    }
                    if (has_edge)
                    {
                        already_matched = true;
                        break;
                    }
                }
            }
            if (already_matched)
                continue;
            std::size_t best = recipients.size();
            float best_distance = std::numeric_limits<float>::infinity();
            for (std::size_t index = 0; index < recipients.size(); ++index)
            {
                if (recipients[index] == target)
                    continue;
                const float value = distance(recipients[index], target);
                if (value < best_distance)
                {
                    best_distance = value;
                    best = index;
                }
            }
            if (best != recipients.size())
                append_unique(forced[best], target);
        }

        std::vector<PointId> pool = outgoing_unique;
        for (PointId candidate : search_candidates)
            append_unique(pool, candidate);

        for (std::size_t index = 0; index < recipients.size(); ++index)
        {
            const PointId owner = recipients[index];
            const std::uint16_t original_degree = get_degree(owner, depth);
            std::uint64_t removed = 0;
            std::vector<PointId> survivors;
            survivors.reserve(original_degree);
            for (std::uint16_t slot = 0; slot < original_degree; ++slot)
            {
                const PointId neighbor = get_neighbor(owner, depth, slot);
                if (neighbor == deleted)
                    ++removed;
                else
                    survivors.push_back(neighbor);
            }
            if (removed == 0 && forced[index].empty())
                continue;
            const std::uint32_t target_degree = std::min<std::uint32_t>(
                capacity_at_depth(depth),
                std::max<std::uint32_t>(original_degree,
                                        static_cast<std::uint32_t>(survivors.size() + forced[index].size())));
            const std::vector<PointId> rewritten = select_repaired_list(owner,
                                                                        depth,
                                                                        survivors,
                                                                        forced[index],
                                                                        pool,
                                                                        target_degree);
            std::uint64_t added = 0;
            for (PointId candidate : rewritten)
            {
                if (std::find(survivors.begin(), survivors.end(), candidate) == survivors.end())
                    ++added;
            }
            record_rewrite(write_neighbor_list(owner, depth, rewritten));
            local.incoming_removed += removed;
            local.replacements_added += added;
            ++local.lists_rewritten;
        }
    }

    void IndexImpl::finalize_delete(PointId id)
    {
        materialize_identity_id_map();
        const ExternalId external = internal_to_external[id];
        internal_to_external[id] = kInvalidPointId;
        --active_points;
        ++stats.deleted_points;
        tree.remove_member_from_path(leaf_ids[id], id);
        if (external < external_to_internal.size())
            external_to_internal[external] = kInvalidPointId;
        reusable_ids.push_back(id);
    }

    bool IndexImpl::erase_sequential(PointId id)
    {
        if (!slot_is_bound(id))
            return false;
        const auto started = std::chrono::steady_clock::now();
        const auto outgoing = collect_outgoing(id);
        DeleteLocalStats total;
        for (std::uint32_t depth = 0; depth < tree.levels; ++depth)
        {
            const auto search_candidates = repair_search(id, depth);
            const auto owners = incoming_owners(id, depth);
            DeleteLocalStats local;
            local.segments = 1;
            local.search_candidates = search_candidates.size();
            local.incoming_owners = owners.size();
            const auto rewrite_started = std::chrono::steady_clock::now();
            repair_delete_depth(id,
                                depth,
                                outgoing[depth],
                                owners,
                                search_candidates,
                                local);
            local.neighbor_rewrite_seconds = std::chrono::duration<double>(
                                                       std::chrono::steady_clock::now() -
                                                       rewrite_started)
                                                       .count();
            local.remaining_edges =
                reverse_adjacency[edge_list_index(id, depth)].size();
            if (local.remaining_edges != 0)
                throw Error("physical delete left an incoming edge to the deleted point");
            total.segments += local.segments;
            total.search_candidates += local.search_candidates;
            total.incoming_owners += local.incoming_owners;
            total.outgoing_removed += local.outgoing_removed;
            total.incoming_removed += local.incoming_removed;
            total.replacements_added += local.replacements_added;
            total.lists_rewritten += local.lists_rewritten;
            total.remaining_edges += local.remaining_edges;
            total.reverse_edge_count_change += local.reverse_edge_count_change;
            total.reverse_updates += local.reverse_updates;
            total.forward_lists_rewritten += local.forward_lists_rewritten;
            total.neighbor_rewrite_seconds += local.neighbor_rewrite_seconds;
        }
        finalize_delete(id);
        commit_neighbor_rewrite_stats(
            {total.reverse_edge_count_change,
             total.reverse_updates,
             total.forward_lists_rewritten},
            total.neighbor_rewrite_seconds);
        stats.delete_segments_processed += total.segments;
        stats.delete_search_candidates += total.search_candidates;
        stats.delete_incoming_owners += total.incoming_owners;
        stats.delete_outgoing_edges_removed += total.outgoing_removed;
        stats.delete_incoming_edges_removed += total.incoming_removed;
        stats.delete_replacement_edges_added += total.replacements_added;
        stats.delete_edge_lists_rewritten += total.lists_rewritten;
        stats.delete_remaining_edges += total.remaining_edges;
        stats.delete_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return true;
    }

    bool IndexImpl::erase(ExternalId external_id)
    {
        prepare_updates();
        const PointId id = to_internal(external_id);
        if (id == kInvalidPointId)
            return false;
        return erase_sequential(id);
    }
}
