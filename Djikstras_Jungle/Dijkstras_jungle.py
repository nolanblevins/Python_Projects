# Nolan Blevins

import heapq
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
# add node
    def add_node(self, node):
        self.nodes.add(node)
        self.edges[node] = []
# add edge from one node to another node
    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node].append((to_node, weight))
        self.edges[to_node].append((from_node, weight))  # This graph is undirected since
                                                         # the times are the same
# implementation of Dijkstra
    def dijkstra(self, start_node):
        distances = {node: float('infinity') for node in self.nodes}
        distances[start_node] = 0
        priority_queue = [(0, start_node)]
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            for neighbor, weight in self.edges[current_node]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
        return distances
# weights of graph in seconds time is always zero to self
weights = {
    # "Base" times are static, so we remove it from the weights
    ("Wolves", "Birds"): 11.2,
    ("Wolves", "Krugs"): 11.0,
    ("Wolves", "Gromp"): 9.3,
    ("Wolves", "Red"): 6.4,
    ("Wolves", "Blue"): 7.1,
    ("Birds", "Krugs"): 9.3,
    ("Birds", "Gromp"): 8.4,
    ("Birds", "Red"): 4.1,
    ("Birds", "Blue"): 9.3,
    ("Krugs", "Gromp"): 18.5,
    ("Krugs", "Red"): 16.1,
    ("Krugs", "Blue"): 12.4,
    ("Gromp", "Red"): 14.2,
    ("Gromp", "Blue"): 3.5,
    ("Red", "Blue"): 20.3,
}

# camps list
camps = ["Wolves", "Birds", "Krugs", "Gromp", "Red", "Blue"]

# print the table
def print_shortest_paths_table(camps, shortest_paths_table):
    print("Shortest paths table (times in seconds):")
    print(f"{'From/To':>10}", end="")
    for camp in camps:
        print(f"{camp:>10}", end="")
    print()

    for start_camp in camps:
        print(f"{start_camp:>10}", end="")
        for end_camp in camps:
            if start_camp != end_camp:
                print(f"{shortest_paths_table[start_camp][end_camp]:>10.1f}", end="")
            else:
                print(f"{'-':>10}", end="")  # no path needed from a camp to itself
        print()

# generate and print the shortest paths table
shortest_paths_table = {camp: {} for camp in camps}
for camp in camps:
    jungle_graph = Graph()
    for node in camps:
        jungle_graph.add_node(node)
    for (start, end), weight in weights.items():
        jungle_graph.add_edge(start, end, weight)
    shortest_paths_table[camp] = jungle_graph.dijkstra(camp)

print_shortest_paths_table(camps, shortest_paths_table)

# initialize the graph and add nodes and edges
jungle_graph = Graph()
for camp in camps:
    jungle_graph.add_node(camp)
for (start, end), weight in weights.items():
    jungle_graph.add_edge(start, end, weight)

optimized_times = {camp: jungle_graph.dijkstra(camp) for camp in camps}
paths = []
original_times_list = []
optimized_times_list = []

# populate the lists with data for each path between camps
for start_camp in camps:
    for end_camp in camps:
        if start_camp != end_camp:
            path_label = f"{start_camp} to {end_camp}"
            paths.append(path_label)
            original_time = weights.get((start_camp, end_camp), weights.get((end_camp, start_camp), float('inf')))
            optimized_time = optimized_times[start_camp].get(end_camp, float('inf'))

            original_times_list.append(original_time)
            optimized_times_list.append(optimized_time)

n = len(paths)
ind = np.arange(n)
width = 0.35       
plt.figure(figsize=(14, 8))
plt.bar(ind, original_times_list, width, label='Original Times')
plt.bar(ind + width, optimized_times_list, width, label='Optimized Times')
plt.xlabel('Paths')
plt.ylabel('Times in seconds')
plt.title('Comparison of Original and Optimized Jungle Path Times')
plt.xticks(ind + width / 2, paths, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
