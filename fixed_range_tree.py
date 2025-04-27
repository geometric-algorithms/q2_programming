import sys
import numpy as np
import numpy.random as random
import time
import math
import matplotlib.pyplot as plt

try:
    sys.setrecursionlimit(10000)
except Exception as e:
    print(f"Warning: Could not set recursion depth limit. {e}")

class RangeTreeNode:
    def __init__(self, split_val=None):
        self.split_val = split_val      
        self.left = None                
        self.right = None              
        self.assoc_struct = None        

class RangeTreeLeaf:
    def __init__(self, points):
        self.points = points if points else []

def build_range_tree(points, dimension, max_dim):
    if not points:
        return None
   
    if len(points) <= 3:
        return RangeTreeLeaf(points)
   
    if dimension == max_dim - 1:
        points_sorted = sorted(points, key=lambda p: p[dimension])
        return build_1d_bst(points_sorted, dimension)
   
    points_sorted = sorted(points, key=lambda p: p[dimension])
    n = len(points_sorted)
   
    median_idx = n // 2
   
    if all(p[dimension] == points_sorted[0][dimension] for p in points_sorted):
        if dimension + 1 < max_dim:
            return build_range_tree(points_sorted, dimension + 1, max_dim)
        else:
            return RangeTreeLeaf(points_sorted)
   
    median_val = points_sorted[median_idx][dimension]
   
    split_idx = median_idx
    while split_idx < n - 1 and points_sorted[split_idx + 1][dimension] == median_val:
        split_idx += 1
   
    if split_idx == n - 1:
        split_idx = n // 2
   
    node = RangeTreeNode(split_val=median_val)
   
    left_points = points_sorted[:split_idx + 1]
    right_points = points_sorted[split_idx + 1:]
   
    if len(left_points) == n and dimension + 1 < max_dim:
        node.assoc_struct = build_range_tree(points_sorted, dimension + 1, max_dim)
        return node
   
    if len(right_points) == n and dimension + 1 < max_dim:
        node.assoc_struct = build_range_tree(points_sorted, dimension + 1, max_dim)
        return node
   
    if left_points:
        node.left = build_range_tree(left_points, dimension, max_dim)
   
    if right_points:
        node.right = build_range_tree(right_points, dimension, max_dim)
   
    if dimension + 1 < max_dim:
        node.assoc_struct = build_range_tree(points_sorted, dimension + 1, max_dim)
   
    return node

def build_1d_bst(points_sorted, dimension):
    n = len(points_sorted)
    if n == 0:
        return None
    if n <= 3:
        return RangeTreeLeaf(points=points_sorted)
   
    median_idx = n // 2
    median_val = points_sorted[median_idx][dimension]
   
    split_idx = median_idx
    while split_idx < n - 1 and points_sorted[split_idx + 1][dimension] == median_val:
        split_idx += 1
   
    if split_idx == n - 1:
        split_idx = n // 2
   
    node = RangeTreeNode(split_val=median_val)
   
    left_points = points_sorted[:split_idx + 1]
    right_points = points_sorted[split_idx + 1:]
   
    if len(left_points) == n:
        return RangeTreeLeaf(points=points_sorted)
   
    if len(right_points) == n:
        return RangeTreeLeaf(points=points_sorted)
   
    node.left = build_1d_bst(left_points, dimension)
    node.right = build_1d_bst(right_points, dimension)
    return node

def query_1d_bst_greater_than(node, val, dimension):
    if node is None:
        return False
   
    if isinstance(node, RangeTreeLeaf):
        return any(point[dimension] > val for point in node.points)
   
    if val < node.split_val:
        return (query_1d_bst_greater_than(node.left, val, dimension) or
                query_1d_bst_greater_than(node.right, val, dimension))
    else:
        return query_1d_bst_greater_than(node.right, val, dimension)

def query_dominator_exists(node, point, dimension, max_dim):
    if node is None:
        return False
   
    if isinstance(node, RangeTreeLeaf):
        for p in node.points:
            if p != point and all(p[i] > point[i] for i in range(max_dim)):
                return True
        return False
   
    if dimension == max_dim - 1:
        return query_1d_bst_greater_than(node, point[dimension], dimension)
   
    if node.split_val is None:
        if hasattr(node, 'assoc_struct') and node.assoc_struct:
            return query_dominator_exists(node.assoc_struct, point, dimension + 1, max_dim)
        return False
   
    p_coord = point[dimension]
   
    if p_coord < node.split_val:
        # 1. Inspect the right child itself - FIXED SECTION
        if node.right:
            if isinstance(node.right, RangeTreeLeaf):
                if any(all(q[i] > point[i] for i in range(max_dim))
                       for q in node.right.points if q != point):
                    return True
            elif getattr(node.right, 'assoc_struct', None):
                if query_dominator_exists(node.right.assoc_struct,
                                          point, dimension + 1, max_dim):
                    return True
            else:  # Internal node without assoc_struct
                if query_dominator_exists(node.right, point, dimension, max_dim):
                    return True
        # 2. No dominator found in right subtree, continue on the left
        return query_dominator_exists(node.left, point, dimension, max_dim)
    else:
        # Defensive improvement: short-circuit when node.right is None
        if node.right is None:
            return False
        return query_dominator_exists(node.right, point, dimension, max_dim)

def brute_force_skyline(points):
    if not points:
        return []
   
    skyline = []
    for p in points:
        is_dominated = False
        for q in points:
            if q != p and all(q[i] > p[i] for i in range(len(p))):
                is_dominated = True
                break
        if not is_dominated:
            skyline.append(list(p))
   
    return skyline

def find_skyline(points, use_brute_force_threshold=25):
    if not points:
        return []
   
    points_tuples = [tuple(p) for p in points]
   
    n = len(points_tuples)
    if n == 1:
        return points
   
    unique_points = list(set(points_tuples))
    n = len(unique_points)
   
    if n <= 1:
        return [list(p) for p in unique_points]
   
    d = len(unique_points[0])
    if d == 0:
        return []
   
    # if n <= use_brute_force_threshold:
    #     return brute_force_skyline(unique_points)
   
    root = build_range_tree(unique_points[:], 0, d)
   
    skyline_result_tuples = []
    for p in unique_points:
        is_dominated = query_dominator_exists(root, p, 0, d)
        if not is_dominated:
            skyline_result_tuples.append(p)
   
    skyline_result = [list(p) for p in skyline_result_tuples]
    return skyline_result

def generate_random_points(n, d):
    return [tuple(random.random() for _ in range(d)) for _ in range(n)]

def run_xd(dim, n):
    st = time.time()
    points_4d = [[np.random.uniform(0, 10) for _ in range(dim)] for _ in range(n)]
    skyline_dom = find_skyline(points_4d)
    skyline_dom.sort()
    et = time.time()
    return et - st