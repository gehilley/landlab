from landlab.components.flow_director.flow_director_d8 import FlowDirectorD8
import numpy as np
from heapq import heappush, heappop
import random

def flood(grid, dx = 1.0, aggradation_slope = 1E-12, fixed_nodes = None):

    pq = []

    rowKernel = np.array([1, 1, 1, 0, 0, -1, -1, -1]).astype(int)
    colKernel = np.array([-1, 0, 1, -1, 1, -1, 0, 1]).astype(int)

    rt2 = np.sqrt(2)
    dxMults = np.array([rt2, 1.0, rt2, 1.0, 1.0, rt2, 1.0, rt2])
    dxs = dx * dxMults

    grid_shape = grid.shape

    def getNeighborIndices(row, col):

        outRows = (rowKernel + int(row))
        outCols = (colKernel + int(col))

        # Determine which indices are out of bounds
        inBounds = (outRows >= 0) * (outRows < grid_shape[0]) * (outCols >= 0) * (outCols < grid_shape[1])
        return (outRows[inBounds], outCols[inBounds], dxMults[inBounds])

    def update_grid_func(elevation):

        def inner_function(row, col, dx):
            if not closed[row, col]:
                grid[row, col] = elevation + aggradation_slope * dx if grid[row,col] <= elevation else grid[row,col]
                closed[row, col] = True
                heappush(pq, (grid[row, col],  (row, col)))

        return inner_function

    closed = np.zeros_like(grid).astype(bool)

    # This is currently set to the top and bottom rows of the model, probably should be generalized:

    if fixed_nodes is None:
        edgeRows = [0 for _ in range(grid.shape[1]-1)]
        edgeRows += [grid.shape[0]-1 for _ in range(grid.shape[1]-1)]
        edgeCols = [i for i in range(grid.shape[0]-1)]
        edgeCols += edgeCols
    else:
        shuffled_fixed_nodes = list(fixed_nodes)
        random.shuffle(shuffled_fixed_nodes)
        [edgeRows, edgeCols] = zip(*shuffled_fixed_nodes)

    for i in range(len(edgeCols)):
        row, col = edgeRows[i], edgeCols[i]
        closed[row, col] = True
        heappush(pq, (grid[row,col], (row, col)))

    counter = 0

    while True:
        try:
            priority, (row, col) = heappop(pq)
            elevation = grid[row, col]
            neighborRows, neighborCols, dxs = getNeighborIndices(row, col)
            list(map(update_grid_func(elevation), neighborRows, neighborCols, dxs))
            counter += 1
        except IndexError:
            break

    return grid

class FlowDirectorD8PriorityFlood(FlowDirectorD8):

    def direct_flow(self):
        (ny, nx) = self._grid.shape
        fixed_nodes = [(i, 0) for (i, edge) in zip(range(ny), self._grid.nodes_at_left_edge) if
                       self._grid.status_at_node[edge] == self._grid.BC_NODE_IS_FIXED_VALUE]
        fixed_nodes += [(i, nx-1) for (i, edge) in zip(range(ny), self._grid.nodes_at_right_edge) if
                       self._grid.status_at_node[edge] == self._grid.BC_NODE_IS_FIXED_VALUE]
        fixed_nodes += [(0, i) for (i, edge) in zip(range(nx), self._grid.nodes_at_bottom_edge) if
                        self._grid.status_at_node[edge] == self._grid.BC_NODE_IS_FIXED_VALUE]
        fixed_nodes += [(ny-1, i) for (i, edge) in zip(range(nx), self._grid.nodes_at_top_edge) if
                        self._grid.status_at_node[edge] == self._grid.BC_NODE_IS_FIXED_VALUE]
        surface_to_flood = flood(self._surface_values.reshape(self._grid.shape), fixed_nodes = fixed_nodes)
        self._surface_values = surface_to_flood.reshape(self._surface_values.shape)
        super().direct_flow()

