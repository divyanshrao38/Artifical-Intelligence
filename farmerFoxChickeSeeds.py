from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import heapq

# Create a more readable graph
def create_graph(solution):
    G = nx.Graph()
    labels = {}
    for i, state in enumerate(solution):
        labels[i] = f"State {i}\n{state}"
        if i > 0:
            prev_state = solution[i - 1]
            action = (
                f"Move {state.farmer - prev_state.farmer} farmer, \n"
                f"{state.fox - prev_state.fox} fox, \n"
                f"{state.chicken - prev_state.chicken} chicken, \n"
                f"{state.seeds - prev_state.seeds} seeds\n"
            )
            G.add_edge(i - 1, i, action=action)

    return G, labels



# Plot the solutions using Matplotlib and NetworkX
def plot_solution(solution, algorithm_name):
    G, labels = create_graph(solution)

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, "action")

    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    node_colors = ["lightblue" for _ in range(len(G))]
    nx.draw(G, pos, node_color=node_colors, node_size=3000, labels=labels, font_size=10)

    # Draw edges
    edge_colors = ["gray" for _ in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.0)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title(f"{algorithm_name} Solution")
    plt.axis('off')  # Turn off axis labels
    plt.show()
# Define the State class to represent the problem state
class State:
    def __init__(self, farmer, fox, chicken, seeds):
        self.farmer = farmer
        self.fox = fox
        self.chicken = chicken
        self.seeds = seeds
        self.cost = farmer + fox + chicken + seeds

    def __str__(self):
        return f"({self.farmer}, {self.fox}, {self.chicken}, {self.seeds})"

    def __eq__(self, other):
        return (
            self.farmer == other.farmer
            and self.fox == other.fox
            and self.chicken == other.chicken
            and self.seeds == other.seeds
        )
    def __lt__(self, other):
        # Compare states based on their cost
        return self.cost < other.cost

    def __hash__(self):
        return hash((self.farmer, self.fox, self.chicken, self.seeds))

# Define the transition rules and valid state checks
def is_valid_state(state):
    # Check if any conflicts exist
    if (state.fox == state.chicken and state.fox != state.farmer) or (
        state.chicken == state.seeds and state.chicken != state.farmer
    ):
        return False
    return True

def get_next_states(current_state):
    next_states = []
    possible_moves = [
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (1, 0, 0, 1),
    ]  # Possible moves with (farmer, fox, chicken, seeds)

    for move in possible_moves:
        new_state = State(
            current_state.farmer ^ move[0],
            current_state.fox ^ move[1],
            current_state.chicken ^ move[2],
            current_state.seeds ^ move[3],
        )
        if is_valid_state(new_state):
            next_states.append(new_state)

    return next_states

# Define the search algorithms
def bfs(initial_state, goal_state):
    visited = set()
    queue = deque([(initial_state, [initial_state])])

    while queue:
        current_state, path = queue.popleft()
        if current_state == goal_state:
            return path

        if current_state not in visited:
            visited.add(current_state)
            for next_state in get_next_states(current_state):
                if next_state not in visited:
                    queue.append((next_state, path + [next_state]))

    return None

def dfs(initial_state, goal_state):
    stack = [(initial_state, [initial_state])]
    visited = set()

    while stack:
        current_state, path = stack.pop()
        if current_state == goal_state:
            return path

        if current_state not in visited:
            visited.add(current_state)
            for next_state in get_next_states(current_state):
                if next_state not in visited:
                    stack.append((next_state, path + [next_state]))

    return None

def heuristic(state, goal_state):
    # Define a heuristic function for A* (here, we use zero heuristic)
    count = (
        (state.farmer == 0) +
        (state.fox == 0) +
        (state.chicken == 0) +
        (state.seeds == 0)
    )
    return count

def a_star(initial_state, goal_state):
    open_set = [(heuristic(initial_state, goal_state), initial_state)]
    came_from = {}
    g_score = {initial_state: 0}

    while open_set:
        open_set.sort(key=lambda x: x[0])
        current_state = open_set.pop(0)[1]

        if current_state == goal_state:
            path = [current_state]
            while current_state in came_from:
                current_state = came_from[current_state]
                path.append(current_state)
            return path[::-1]

        for next_state in get_next_states(current_state):
            tentative_g_score = g_score[current_state] + 1
            if (
                next_state not in g_score
                or tentative_g_score < g_score[next_state]
            ):
                came_from[next_state] = current_state
                g_score[next_state] = tentative_g_score
                f_score = tentative_g_score + heuristic(next_state, goal_state)
                open_set.append((f_score, next_state))

    return None

def dijkstra(initial_state, goal_state):
    open_set = [(0, initial_state)]
    came_from = {}
    cost_so_far = {initial_state: 0}

    while open_set:
        current_cost, current_state = heapq.heappop(open_set)

        if current_state == goal_state:
            path = [current_state]
            while current_state in came_from:
                current_state = came_from[current_state]
                path.append(current_state)
            return path[::-1]

        for next_state in get_next_states(current_state):
            new_cost = cost_so_far[current_state] + 1
            if (
                next_state not in cost_so_far
                or new_cost < cost_so_far[next_state]
            ):
                cost_so_far[next_state] = new_cost
                priority = new_cost
                heapq.heappush(open_set, (priority, next_state))
                came_from[next_state] = current_state

    return None
# Define initial and goal states
initial_state = State(0, 0, 0, 0)
goal_state = State(1, 1, 1, 1)

# Solve using BFS
bfs_solution = bfs(initial_state, goal_state)

# Solve using DFS
dfs_solution = dfs(initial_state, goal_state)

# Solve using A*
a_star_solution = a_star(initial_state, goal_state)

dijkstra_solution = dijkstra(initial_state, goal_state)

# Print solutions
print("BFS Solution:")
if bfs_solution:
    for state in bfs_solution:
        print(state)
    plot_solution(bfs_solution, "BFS")
else:
    print("No solution found.")

print("\nDFS Solution:")
if dfs_solution:
    for state in dfs_solution:
        print(state)
    plot_solution(dfs_solution, "DFS")
else:
    print("No solution found.")

print("\nA* Solution:")
if a_star_solution:
    for state in a_star_solution:
        print(state)
    plot_solution(a_star_solution, "A*")
else:
    print("No solution found.")

# Print solutions
print("Dijkstra's Solution:")
if dijkstra_solution:
    for state in dijkstra_solution:
        print(state)
    plot_solution(dijkstra_solution, "Dijkstra's")
else:
    print("No solution found.")
