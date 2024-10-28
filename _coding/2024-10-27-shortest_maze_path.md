---
layout: post
title: "Shortest Path in a Maze"
katex: False
blurb: ""
img: ""
author: "Malav Patel"
categories: journal
tags: []
<!-- image: -->
---

You are given a grid of size `m` by `n`, a 2d integer array `walls`, and a 2-tuple `escape_point`. A maze runner begins at position (0, 0). The runner can move up, down, left, right in the grid. Their goal is to reach the escape point. However, there are walls in the maze (their 2d locations specified by `walls`) that are impassable. Find the minimum number of steps needed to get to the escape point. If there is no path, return -1.

## Solution
The idea is to use Breadth-First-Search. It works because BFS will sweep all the closest neighbors starting at the start point. If they happen to be the escape, the program terminates. Otherwise, we look at the neighbors' neighbors and so on.

#### Pseudocode
```
def bfs_maze(m, n, walls, start_point, escape_point):
    q = create_empty_queue()
    visited = create_empty_set()
    q.put((start_point, 0)) # add (start_point, level in bfs tree)

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    while not q.empty():
        pos, level = queue.get_item()
        visited.add(pos)

        if pos == escape_point:
            return level
        
        for dir in directions:
            neighbor = pos + direction

            if (neighbor not a wall) and (neighbor not visited) and (neighbor in bounds):
                q.add( (neighbor, level + 1) )
    
    return -1 # return -1 if no valid path found
```

### Python
```python

def bfs_maze(m, n, walls, escape_point):
    q = queue.Queue() # create an empty queue to store to-be-visited positions in the grid
    visited = set() # create an empty set to hold already visited positions in the grid

    start = ((0, 0), 0) # set start point of maze at (0, 0) and level 0 of bfs tree
    q.put(start) # place start point and level into queue

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)] # possible movement directions

    while not q.empty():
        pos, level = q.get()
        visited.add(pos)

        if pos == escape_point:
            return level

        for dir in directions:
            neighbor = (pos[0] + dir[0], pos[1]+ dir[1])

            if (neighbor not in walls) and (neighbor not in visited) and valid_point(neighbor, m, n):
                q.put((neighbor, level + 1))

    return -1

def valid_point(pos, m, n):
    i, j = pos

    if 0 <= i < m and 0 <= j < n:
        return True
    else:
        return False
```

#### Example

Consider the following maze where solid squares are walls and the `*` is the escape point:

```bash
 .  .  .  .  . 
 ■  ■  .  ■  ■ 
 .  .  .  .  . 
 .  .  .  .  . 
 .  .  .  .  * 
```
One of shortest paths has 8 steps (there are several with 8 steps) and looks like the following:
```bash
 +  +  +  .  . 
 ■  ■  +  ■  ■ 
 .  .  +  +  + 
 .  .  .  .  + 
 .  .  .  .  * 
```
The output of our program is as expected:
```python
rows = 5
cols = 5
walls = [(1, 0), (1, 1), (1, 3), (1, 4)]
escape_point = (4, 4)
bfs_maze(rows, cols, walls, escape_point)

8 # output
```




