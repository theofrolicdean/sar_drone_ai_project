import numpy as np
import heapq
import time
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import csv
import argparse

GRID_WIDTH = 25
GRID_HEIGHT = 25
GRID_DEPTH = 10
COLOR_OBSTACLE = "tomato"
COLOR_HOME = "limegreen"
COLOR_SURVIVOR = "gold"
COLOR_SPIRAL_PATH = "cyan"
COLOR_RETURN_PATH = "fuchsia"

class DroneSimulation3D:
    def __init__(self, width, height, depth, num_obstacles, num_survivors, drone_speed):
        self.width = width
        self.height = height
        self.depth = depth
        self.num_obstacles = num_obstacles
        self.num_survivors = num_survivors
        self.drone_speed = drone_speed # cell/second
        self.grid, self.obstacle_heights = self._create_grid()
        self.home_pos = self._place_item(is_home=True)
        self.survivor_pos = [self._place_item() for _ in range(self.num_survivors)]
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.evaluation_data = []

    def _create_grid(self):
        grid = np.zeros((self.height, self.width), dtype=int)
        heights = {}
        for _ in range(self.num_obstacles):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            grid[y, x] = 1 # 1 -> obstacle
            heights[(x, y)] = random.randint(1, self.depth)
        return grid, heights

    def _is_valid_position(self, pos):
        return self.grid[pos[1], pos[0]] == 0

    def _place_item(self, is_home=False):
        existing_pos = {self.home_pos[:2]} if hasattr(self, 'home_pos') and self.home_pos else set()
        if hasattr(self, 'survivor_pos'):
            existing_pos.update(s[:2] for s in self.survivor_pos)

        while True:
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if self._is_valid_position(pos) and pos not in existing_pos:
                return (pos[0], pos[1], 0)

    def run_simulation(self, show_animation=True):
        print("--- Starting 3D Drone Rescue Simulation ---")
        print(f"Grid Size: {self.width}x{self.height}")
        print(f"Obstacles: {self.num_obstacles}")
        print(f"Survivors: {self.num_survivors}")
        print(f"Home Base: {self.home_pos}")
        print(f"Drone Speed: {self.drone_speed} cells/sec")
        print("----------------------------------------")

        current_pos = self.home_pos
        for i, survivor in enumerate(self.survivor_pos):
            print(f"\n--- Mission: Find Survivor {i+1} at {survivor[:2]} ---")
            
            spiral_path = list(self.spiral_search(current_pos, survivor))
            search_duration_s = len(spiral_path) / self.drone_speed
            print(f"  -> Found survivor via spiral search (path length: {len(spiral_path)}, duration: {search_duration_s:.2f}s)")
            algorithms = {
                "A*": PathFinder.a_star,
                "UCS": PathFinder.uniform_cost_search,
                "Greedy": PathFinder.greedy_best_first_search
            }
            for name, func in algorithms.items():
                print(f"  -> Planning return with {name}")
                pathfinder = PathFinder(self.grid)
                
                start_time = time.perf_counter()
                return_path_2d = pathfinder.run_search(func, survivor[:2], self.home_pos[:2])
                end_time = time.perf_counter()
                compute_time_ms = (end_time - start_time) * 1000
                
                # Format the spiral path for CSV logging
                spiral_path_str = ";".join([f"{p[0]},{p[1]}" for p in spiral_path])
                
                run_data = {
                    "survivor_id": i + 1,
                    "algorithm": name,
                    "search_path_cost": len(spiral_path),
                    "search_duration_s": search_duration_s,
                    "compute_time_ms": compute_time_ms,
                    "search_path_coords": spiral_path_str
                }

                if return_path_2d:
                    flight_altitude = self.depth + 2
                    return_path_3d = [(p[0], p[1], flight_altitude) for p in return_path_2d]
                    return_duration_s = len(return_path_3d) / self.drone_speed
                    total_mission_duration_s = search_duration_s + return_duration_s
                    
                    print(f"     Path found with cost: {len(return_path_3d)} in {compute_time_ms:.2f} ms. Flight duration: {return_duration_s:.2f}s")

                    return_path_str = ";".join([f"{p[0]},{p[1]}" for p in return_path_2d])

                    run_data.update({
                        "success": True,
                        "return_path_cost": len(return_path_3d),
                        "return_duration_s": return_duration_s,
                        "total_path_cost": len(spiral_path) + len(return_path_3d),
                        "total_mission_duration_s": total_mission_duration_s,
                        "return_path_coords": return_path_str
                    })
                    
                    if show_animation:
                        full_mission_path = spiral_path + return_path_3d
                        self._visualize_mission(full_mission_path, survivor, name, i + 1)
                else:
                    print(f"     NO RETURN PATH COULD BE FOUND in {compute_time_ms:.2f} ms.")
                    run_data.update({
                        "success": False,
                        "return_path_cost": None,
                        "return_duration_s": None,
                        "total_path_cost": None,
                        "total_mission_duration_s": None,
                        "return_path_coords": None
                    })
                
                self.evaluation_data.append(run_data)
            
            current_pos = self.home_pos

        print("\n--- All survivors rescued. Simulation Complete. ---")
        plt.close('all')
        self._save_evaluation_results()

    def spiral_search(self, start, goal):
        """Generates points in an outward spiral path from start until goal is reached."""
        x, y, z = start
        flight_altitude = self.depth + 2
        yield (x, y, flight_altitude)
        
        dx, dy = 1, 0
        steps_in_segment = 1
        segment_passed = 0
        max_steps = self.width * self.height * 2 
        steps_taken = 0

        while (x, y) != goal[:2] and steps_taken < max_steps:
            for _ in range(steps_in_segment):
                x, y = x + dx, y + dy
                steps_taken += 1
                if (x, y) == goal[:2]:
                    yield (x, y, flight_altitude)
                    return
                if 0 <= x < self.width and 0 <= y < self.height:
                    yield (x, y, flight_altitude)
            dx, dy = -dy, dx
            segment_passed += 1
            if segment_passed == 2:
                steps_in_segment += 1
                segment_passed = 0

    def _setup_3d_plot(self, title):
        """Sets up the initial state of the 3D plot."""
        self.ax.clear()
        self.ax.set_title(title, fontsize=16)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_zlabel('Z (Altitude)')
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_zlim(0, self.depth + 5)

        for (x, y), height in self.obstacle_heights.items():
            self.ax.bar3d(x, y, 0, 1, 1, height, color=COLOR_OBSTACLE, alpha=0.7)

        self.ax.scatter(*self.home_pos, color=COLOR_HOME, s=150, marker='H', label='Home', depthshade=False)
        for i, s_pos in enumerate(self.survivor_pos):
             self.ax.scatter(*s_pos, color=COLOR_SURVIVOR, s=150, marker='*', label=f'Survivor {i+1}', depthshade=False)
        self.ax.legend()

    def _visualize_mission(self, path, survivor, algo_name, survivor_num):
        """Animates a full mission path in 3D."""
        title = f"{algo_name} | Rescuing Survivor {survivor_num}"
        self._setup_3d_plot(title)
        
        survivor_3d_pos = (survivor[0], survivor[1], self.depth + 2)
        try:
            spiral_len = path.index(survivor_3d_pos) + 1
        except ValueError:
            spiral_len = len(list(self.spiral_search(self.home_pos, survivor)))

        line1, = self.ax.plot([], [], [], color=COLOR_SPIRAL_PATH, lw=2, label='Spiral Search')
        line2, = self.ax.plot([], [], [], color=COLOR_RETURN_PATH, lw=2, label='Return Path')
        drone_marker, = self.ax.plot([], [], [], 'ro', markersize=8, label='Drone')

        def update(frame):
            current_path = path[:frame+1]
            x_data = [p[0] for p in current_path]
            y_data = [p[1] for p in current_path]
            z_data = [p[2] for p in current_path]

            if frame < spiral_len:
                line1.set_data(x_data, y_data)
                line1.set_3d_properties(z_data)
            else:
                spiral_x = [p[0] for p in path[:spiral_len]]
                spiral_y = [p[1] for p in path[:spiral_len]]
                spiral_z = [p[2] for p in path[:spiral_len]]
                line1.set_data(spiral_x, spiral_y)
                line1.set_3d_properties(spiral_z)
                
                return_x = [p[0] for p in path[spiral_len-1:frame+1]]
                return_y = [p[1] for p in path[spiral_len-1:frame+1]]
                return_z = [p[2] for p in path[spiral_len-1:frame+1]]
                line2.set_data(return_x, return_y)
                line2.set_3d_properties(return_z)

            drone_marker.set_data([x_data[-1]], [y_data[-1]])
            drone_marker.set_3d_properties([z_data[-1]])
            
            return line1, line2, drone_marker

        anim = FuncAnimation(self.fig, update, frames=len(path), blit=False, interval=50)
        plt.show()

    def _save_evaluation_results(self):
        """Saves the collected evaluation data to a CSV file and plots it."""
        if not self.evaluation_data:
            print("No evaluation data to save.")
            return

        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, 'evaluation_results.csv')
        headers = [
            "survivor_id", "algorithm", "success", "search_path_cost", "return_path_cost",
            "total_path_cost", "search_duration_s", "return_duration_s", "total_mission_duration_s",
            "compute_time_ms", "search_path_coords", "return_path_coords"
        ]
        with open(csv_path, 'w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.evaluation_data)
        print(f"\nRaw evaluation data saved to {csv_path}")


        self._create_and_save_summary(results_dir)
        self._plot_evaluation_results()

    def _create_and_save_summary(self, results_dir):
        """Calculates and saves a statistical summary of the evaluation data."""
        results_by_algo = self._get_results_by_algo()
        summary_data = []
        for algo, data in results_by_algo.items():
            costs = data['costs']
            times = data['times']
            durations = data['durations']
            
            summary_data.append({
                'algorithm': algo,
                'num_successful_runs': len(costs),
                'mean_cost': np.mean(costs) if costs else 0,
                'median_cost': np.median(costs) if costs else 0,
                'std_dev_cost': np.std(costs) if costs else 0,
                'min_cost': np.min(costs) if costs else 0,
                'max_cost': np.max(costs) if costs else 0,
                'mean_compute_ms': np.mean(times) if times else 0,
                'median_compute_ms': np.median(times) if times else 0,
                'std_dev_compute_ms': np.std(times) if times else 0,
                'mean_duration_s': np.mean(durations) if durations else 0,
                'median_duration_s': np.median(durations) if durations else 0,
                'std_dev_duration_s': np.std(durations) if durations else 0,
            })
            
        summary_csv_path = os.path.join(results_dir, 'evaluation_summary.csv')
        if not summary_data: return
        summary_headers = summary_data[0].keys()
        with open(summary_csv_path, 'w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=summary_headers)
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"Statistical summary saved to {summary_csv_path}")

    def _get_results_by_algo(self):
        results_by_algo = {}
        for row in self.evaluation_data:
            algo = row['algorithm']
            if algo not in results_by_algo:
                results_by_algo[algo] = {'costs': [], 'times': [], 'durations': []}
            if row['success']:
                results_by_algo[algo]['costs'].append(row['return_path_cost'])
                results_by_algo[algo]['times'].append(row['compute_time_ms'])
                results_by_algo[algo]['durations'].append(row['total_mission_duration_s'])
        return results_by_algo

    def _plot_evaluation_results(self):
        if not self.evaluation_data: return

        labels = [f"S{r['survivor_id']}-{r['algorithm']}" for r in self.evaluation_data]
        costs = [r.get('return_path_cost', 0) for r in self.evaluation_data]
        times = [r.get('compute_time_ms', 0) for r in self.evaluation_data]
        durations = [r.get('total_mission_duration_s', 0) for r in self.evaluation_data]

        color_map = {
            "A*": '#ff79c6', 
            "UCS": '#8be9fd', 
            "Greedy": '#50fa7b'
        }
        colors = [color_map.get(r['algorithm'], '#f8f8f2') for r in self.evaluation_data]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle('Raw Performance per Mission', fontsize=16)

        ax1.bar(labels, costs, color=colors)
        ax1.set_title('Return Path Cost')
        ax1.set_ylabel('Number of Steps')
        ax1.tick_params(axis='x', rotation=90)

        ax2.bar(labels, times, color=colors)
        ax2.set_title('Compute Time')
        ax2.set_ylabel('Time (ms)')
        ax2.tick_params(axis='x', rotation=90)
        
        ax3.bar(labels, durations, color=colors)
        ax3.set_title('Total Mission Duration')
        ax3.set_ylabel('Duration (seconds)')
        ax3.tick_params(axis='x', rotation=90)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = os.path.join('results', 'evaluation_plot.png')
        plt.savefig(plot_path)
        print(f"Evaluation plot saved to {plot_path}")
        plt.close(fig)



#path finding
class PathFinder:
    def __init__(self, grid):
        self.grid = grid
        self.height, self.width = grid.shape

    def run_search(self, search_func, start, goal):
        """Helper to run a search and return the path."""
        came_from = search_func(self, start, goal)
        if came_from:
            path = self._reconstruct_path(came_from, goal)
            if path and path[0] == start:
                return path
        return None

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == 0:
                cost = 1
                neighbors.append(((nx, ny), cost))
        return neighbors

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from and came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self, start, goal):
        open_set = [(0, start)]
        came_from = {start: None}
        g_score = { (x,y): float('inf') for x in range(self.width) for y in range(self.height) }
        g_score[start] = 0
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal: return came_from
            
            for neighbor, cost in self.get_neighbors(current):
                tentative_g_score = g_score[current] + cost
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return None

    def uniform_cost_search(self, start, goal):
        open_set = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal: return came_from
            
            for neighbor, cost in self.get_neighbors(current):
                new_cost = cost_so_far[current] + cost
                if new_cost < cost_so_far.get(neighbor, float('inf')):
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (new_cost, neighbor))
        return None

    def greedy_best_first_search(self, start, goal):
        open_set = [(0, start)]
        came_from = {start: None}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal: return came_from
            
            for neighbor, _ in self.get_neighbors(current):
                if neighbor not in came_from:
                    priority = self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
        return None



def main():
    """Parses command-line arguments and runs the simulation."""
    parser = argparse.ArgumentParser(description="Run a 3D drone rescue simulation with pathfinding algorithm evaluation.")
    parser.add_argument('--survivors', type=int, default=random.randint(1, 3),
                        help='Number of survivors to place in the simulation.')
    parser.add_argument('--obstacles', type=int, default=random.randint(40, 80),
                        help='Number of obstacles to place in the simulation.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility.')
    parser.add_argument('--speed', type=float, default=5.0,
                        help='Simulated drone speed in cells per second.')
    parser.add_argument('--no-animation', action='store_true',
                        help='Run in batch mode without showing 3D animations to speed up evaluation.')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    sim = DroneSimulation3D(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        depth=GRID_DEPTH,
        num_obstacles=args.obstacles,
        num_survivors=args.survivors,
        drone_speed=args.speed
    )
    sim.run_simulation(show_animation=not args.no_animation)


if __name__ == '__main__':
    main()
