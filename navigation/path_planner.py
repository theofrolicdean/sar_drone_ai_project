import networkx as nx
from utils.logger import logger

# Memakai A* algorithm

class PathPlanner:
    def __init__(self, grid_size_m, cell_size_cm):
        self.grid_size = int((grid_size_m * 100) / cell_size_cm)
        self.cell_size_cm = cell_size_cm
        self.graph = self._create_grid_graph()

    def _create_grid_graph(self):
        G = nx.grid_2d_graph(self.grid_size, self.grid_size)
        return G

    def _world_to_grid(self, pos_cm):
        # cm -> to grid coordinates
        grid_x = int((pos_cm[0] + (self.grid_size * self.cell_size_cm) / 2) / self.cell_size_cm)
        grid_y = int((pos_cm[1] + (self.grid_size * self.cell_size_cm) / 2) / self.cell_size_cm)
        return max(0, min(self.grid_size - 1, grid_x)), max(0, min(self.grid_size - 1, grid_y))

    def plan_path(self, start_pos_cm, goal_pos_cm):
        start_grid = self._world_to_grid(start_pos_cm)
        goal_grid = self._world_to_grid(goal_pos_cm)

        logger.info(f"Planning path from {start_grid} to {goal_grid}")

        try:
            path_grid = nx.astar_path(self.graph, start_grid, goal_grid, heuristic=self._heuristic)
            # Convert grid path to cm
            path_world = [((x * self.cell_size_cm) - (self.grid_size * self.cell_size_cm / 2),
                    (y * self.cell_size_cm) - (self.grid_size * self.cell_size_cm / 2)) 
                for x, y in path_grid]
            logger.info(f"Path found with {len(path_world)} waypoints.")
            return path_world
        except nx.NetworkXNoPath:
            logger.error(f"No path found")
            return None

    @staticmethod
    def _heuristic(a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])