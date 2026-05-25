#!/usr/bin/env python3
"""Create pairwise_similarity.csv from k-hop graph neighborhood overlap."""

import argparse
import csv
from collections import defaultdict, deque


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_graph(path, source_col, target_col, directed):
    graph = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row[source_col]
            target = row[target_col]
            graph[source].add(target)
            graph.setdefault(target, set())
            if not directed:
                graph[target].add(source)
    return graph


def khop_nodes(graph, start, k):
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= k:
            continue
        for neighbor in graph.get(node, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
    seen.discard(start)
    return seen


def jaccard(left, right):
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-col", default="source")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--node-col", default="node_id")
    args = parser.parse_args()

    if args.k < 1:
        raise SystemExit("--k must be at least 1")

    graph = read_graph(args.edges, args.source_col, args.target_col, args.directed)
    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    neighborhood_cache = {}

    def neighborhood(node):
        if node not in neighborhood_cache:
            neighborhood_cache[node] = khop_nodes(graph, node, args.k)
        return neighborhood_cache[node]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_neighborhood = neighborhood(eval_row[args.node_col])
            for train_row in train_rows:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_row[args.train_id_col],
                        "similarity": jaccard(eval_neighborhood, neighborhood(train_row[args.node_col])),
                    }
                )


if __name__ == "__main__":
    main()
