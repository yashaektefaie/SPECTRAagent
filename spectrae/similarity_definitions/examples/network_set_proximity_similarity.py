#!/usr/bin/env python3
"""Create pairwise_similarity.csv from shortest-path proximity between entity sets."""

import argparse
import csv
import math
from collections import defaultdict, deque


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_set(value, delimiter):
    return {token.strip() for token in str(value).split(delimiter) if token.strip()}


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


def distances_from_sources(graph, sources):
    distances = {}
    queue = deque()
    for source in sources:
        distances[source] = 0
        queue.append(source)
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, set()):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def mean_min_distance(graph, left, right, unreachable_distance):
    if not left and not right:
        return 0.0
    if not left or not right:
        return float(unreachable_distance)
    right_distances = distances_from_sources(graph, right)
    left_distances = distances_from_sources(graph, left)
    left_to_right = [right_distances.get(node, unreachable_distance) for node in left]
    right_to_left = [left_distances.get(node, unreachable_distance) for node in right]
    return (sum(left_to_right) / len(left_to_right) + sum(right_to_left) / len(right_to_left)) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-col", default="source")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--set-col", default="entities")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--unreachable-distance", type=float, default=10.0)
    args = parser.parse_args()

    if args.scale <= 0.0:
        raise SystemExit("--scale must be positive")

    graph = read_graph(args.edges, args.source_col, args.target_col, args.directed)
    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    train_sets = [
        (row[args.train_id_col], parse_set(row[args.set_col], args.delimiter)) for row in train_rows
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_set = parse_set(eval_row[args.set_col], args.delimiter)
            for train_id, train_set in train_sets:
                distance = mean_min_distance(
                    graph,
                    eval_set,
                    train_set,
                    args.unreachable_distance,
                )
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": math.exp(-distance / args.scale),
                    }
                )


if __name__ == "__main__":
    main()
