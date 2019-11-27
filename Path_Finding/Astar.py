import numpy as np

class Node:
    def __init__(self, x, y, walkable, tag = None, rCost = 0):
        self.x = x
        self.y = y
        self.gCost = 0
        self.hCost = 0
        self.fCost = 0
        self.rCost = rCost
        self.walkable = walkable
        if tag is None:
            if walkable:
                self.tag = 'PATH'
            else:
                self.tag = 'WALL'
        else:
            self.tag = tag
        self.Parent = None #linking for retracing the path
    def compareTo(self, node2):
        return self.x==node2.x and self.y==node2.y
    def printSelf(self):
        print(' <', self.x, self.y, self.walkable, [self.gCost, self.hCost, self.fCost], self.tag, self.rCost, self.Parent, '>')
    def get(self, tag):
        dict_ = {'x':self.x, 'y':self.y, 'walkable':(0 if self.walkable else 1), 'hCost':self.hCost, 'gCost':self.gCost, 'fCost':self.fCost, 'rCost': self.rCost,'index':(self.x+self.y/10), 'tag':self.tag}
        return dict_[tag]
    def copy(self):
        return Node(self.x, self.y, self.walkable, tag=self.tag, rCost=self.rCost)

class Astar:
    def __init__(self, MAX_HEAT_MAP_VALUE=1, HEAT_MAP_WEIGHT=1):
        self.MAX_HEAT_MAP_VALUE = MAX_HEAT_MAP_VALUE
        self.HEAT_MAP_WEIGHT = HEAT_MAP_WEIGHT

    def getHDistance(self, nodeA, nodeB, APPLY_HEAT = False):
        dx = nodeA.x - nodeB.x
        dy = nodeA.y - nodeB.y
        # costVal = np.sqrt(dx*dx + dy*dy)
        costVal = np.abs(dx)+np.abs(dy)
        # testing * (1+delta resistance cost)
        if APPLY_HEAT:
            drCost = (nodeB.rCost - nodeA.rCost)
            costVal = costVal*(1 + drCost/self.MAX_HEAT_MAP_VALUE*self.HEAT_MAP_WEIGHT)
        return costVal

    def retracePath(self, sNode, eNode):
        path = []
        cur_node = eNode
        loop = True
        path.append(cur_node)
        while loop:
            cur_node = cur_node.Parent
            loop = (cur_node.compareTo(sNode) == False)
            path.insert(0, cur_node)

        return path
        
    def getNeighbours(self, cnode, grid, ALLOW_DIAG):
        neighbours = []
        for i in range(cnode.y-1, cnode.y+2):
            for j in range(cnode.x-1, cnode.x+2):
                if not ALLOW_DIAG:
                    if (j-cnode.x)*(i-cnode.y) != 0: # skip diagonal
                        continue
                if (i>= 0 and i<len(grid) and j>=0 and j<len(grid[0]) and grid[i][j].walkable):
                    neighbours.append(grid[i][j].copy()) # make a copy
        return neighbours
        
    def getPathDistance(self, path):
        return path[len(path)-1].fCost

    def findPath(self, world, sNode, eNode, ALLOW_DIAG=True):
        openSet = []
        closeSet = []
        
        if sNode.walkable:
            sNode.gCost = 0
            sNode.hCost = self.getHDistance(sNode, eNode, APPLY_HEAT=False)
            sNode.fCost = sNode.hCost
        
            openSet.append(sNode)
            
            # self.printRow(openSet, 'index', 'Openset:')

            while(len(openSet)>0):
                sIt = openSet[0]
                for it in openSet:
                    if it.fCost < sIt.fCost or (it.fCost == sIt.fCost and it.hCost <= sIt.hCost):
                        sIt = it
                        
                closeSet.append(sIt)
                openSet.remove(sIt)
                # self.printRow(openSet, 'index', 'Openset:')
                # self.printRow(closeSet, 'index', 'closeSet:')
                sIt = closeSet[len(closeSet)-1]

                neighbours = self.getNeighbours(sIt, world, ALLOW_DIAG)
                # self.printRow(neighbours, 'index', 'Neighbour:')

                for it in neighbours:
                    isUnique = True
                    for it2 in closeSet:
                        if it.compareTo(it2):
                            isUnique = False
                    if isUnique:
                        gCostTemp = sIt.gCost + self.getHDistance(sIt, it, APPLY_HEAT=True)
                        for it2 in openSet:
                            if it.compareTo(it2):
                                isUnique = False
                        if isUnique:
                            it.gCost = gCostTemp
                            it.Parent = sIt
                            it.hCost = self.getHDistance(it, eNode, APPLY_HEAT=False)
                            it.fCost = it.gCost + it.hCost
                            openSet.append(it)
                        elif it.gCost > gCostTemp:
                            it.gCost = gCostTemp
                            it.Parent = sIt
                            it.hCost = self.getHDistance(it, eNode)
                            it.fCost = it.gCost + it.hCost

                if sIt.compareTo(eNode):
                    return self.retracePath(sNode, sIt)
    
        return None;

    def extractPath(self, Astar_Path):
        path = []
        for node in Astar_Path:
            path.append((node.x, node.y))
        return path

    def printRow(self, list, tag, title=None):
        row = []
        for item in list:
            row.append(item.get(tag))
        if title is not None:
            print(title, tag, row)
        else:
            print(tag, row)

    def printAll2D(self, objs, tag):
        for row in objs:
            self.printRow(row, tag)

    def genMap(self, mapInput, sPos, ePos, PATH_VALUE = 0, heat_map = None):
        grid = []
        for y in range(0, len(mapInput)):
            row = []
            for x in range(0, len(mapInput[0])):
                if heat_map is not None:
                    newnode = Node(x, y, (mapInput[y][x] == PATH_VALUE), rCost=heat_map[y][x])
                else:
                    newnode = Node(x, y, (mapInput[y][x] == PATH_VALUE))
                row.append(newnode)
            grid.append(row)
        
        sNode = grid[sPos[1]][sPos[0]]
        eNode = grid[ePos[1]][ePos[0]]
        sNode.tag = "START"
        eNode.tag = "END"
        return grid, sNode, eNode

if __name__ == '__main__':
    mapp = [[1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
            ]
    ASTAR = Astar()
    map_grid, sNode, eNode  = ASTAR.genMap(mapp, (1,1), (3,4))
    ASTAR.printAll2D(map_grid, 'walkable')
    sNode.printSelf()
    eNode.printSelf()
    print('--- RESULT ---')
    path = ASTAR.findPath(map_grid, sNode, eNode, ALLOW_DIAG=False)
    print(ASTAR.extractPath(path))
    ASTAR.printRow(path, 'index')
    ASTAR.printRow(path, 'fCost')
    ASTAR.printRow(path, 'gCost')
    ASTAR.printRow(path, 'hCost')
    ASTAR.printRow(path, 'rCost')
    ASTAR.printRow(path, 'tag')
