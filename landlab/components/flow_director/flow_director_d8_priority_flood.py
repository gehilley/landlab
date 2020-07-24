from landlab.components.flow_director.flow_director_d8 import FlowDirectorD8
import numpy as np
import heapq
import random

def flood(grid, dx = 1.0, aggradation_slope = 1E-12, fixed_nodes = None):

    class priorityQueue:
        # Implements a priority queue using heapq. Python has a priority queue module built in, but it
        # is not stably sorted (meaning that two items who are tied in priority are treated arbitrarily, as opposed to being
        # returned on a first in first out basis). This circumvents that by keeping a count on the items inserted and using that
        # count as a secondary priority

        def __init__(self):
            # A counter and the number of items are stored separately to ensure that items remain stably sorted and to
            # keep track of the size of the queue (so that we can check if its empty, which will be useful will iterating
            # through the queue)
            self.__pq = []
            self.__counter = 0
            self.__nItems = 0

        def get(self):
            # Remove an item and its priority from the queue
            priority, count, item = heapq.heappop(self.__pq)
            self.__nItems -= 1
            return priority, item

        def put(self, priority, item):
            # Add an item to the priority queue
            self.__counter += 1
            self.__nItems += 1
            entry = [priority, self.__counter, item]
            heapq.heappush(self.__pq, entry)

        def isEmpty(self):
            return self.__nItems == 0

    def getNeighborIndices(row, col, grid_shape):
        # Search kernel for D8 flow routing, the relative indices of each of the 8 points surrounding a pixel
        rowKernel = np.array([1, 1, 1, 0, 0, -1, -1, -1])
        colKernel = np.array([-1, 0, 1, -1, 1, -1, 0, 1])

        rt2 = np.sqrt(2)
        dxMults = np.array([rt2, 1.0, rt2, 1.0, 1.0, rt2, 1.0, rt2])  # Unit Distance from pixel to surrounding coordinates

        # Find all the surrounding indices
        outRows = (rowKernel + row).astype(int)
        outCols = (colKernel + col).astype(int)

        # Determine which indices are out of bounds
        inBounds = (outRows >= 0) * (outRows < grid_shape[0]) * (outCols >= 0) * (outCols < grid_shape[1])
        return (outRows[inBounds], outCols[inBounds], dxMults[inBounds])

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

    priority_queue = priorityQueue()

    for i in range(len(edgeCols)):
        row, col = edgeRows[i], edgeCols[i]

        closed[row, col] = True
        priority_queue.put(grid[row, col], (row, col))

    while not priority_queue.isEmpty():

        priority, (row, col) = priority_queue.get()
        elevation = grid[row, col]

        neighborRows, neighborCols, dxMults = getNeighborIndices(row, col, grid.shape)
        dxs = dx * dxMults

        for i in range(len(neighborCols)):
            should_fill = True

            if not closed[neighborRows[i], neighborCols[i]]:
                # If this was a hole (lower than the cell downstream), fill it
                if grid[neighborRows[i], neighborCols[i]] <= elevation:
                    if should_fill:
                        grid[neighborRows[i], neighborCols[i]] = elevation + aggradation_slope * dxs[i]

                closed[neighborRows[i], neighborCols[i]] = True
                if should_fill:
                    priority_queue.put(grid[neighborRows[i], neighborCols[i]], [neighborRows[i], neighborCols[i]])

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

