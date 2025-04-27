import bisect
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from fixed_range_tree import find_skyline

def compute_3d_skyline(points):
    if not points:
        return []
   
    points_sorted = sorted(points, key=lambda p: (-p[2], p[0], p[1]))
   
    skyline_2d = []
    result = []
   
    for p in points_sorted:
        x, y, z = p
       
        idx = bisect.bisect_right(skyline_2d, (x, float('inf')))
       
        dominated = False
        if idx < len(skyline_2d):
            if skyline_2d[idx][1] > y:
                dominated = True
       
        if not dominated:
            pos = bisect.bisect_left(skyline_2d, (x, y))
            skyline_2d.insert(pos, (x, y))
           
            j = pos - 1
            while j >= 0:
                if skyline_2d[j][1] <= y:
                    del skyline_2d[j]
                    j -= 1
                else:
                    break
           
            result.append(p)
   
    return result

def generate_random_3d_points(n):
    return [[np.random.uniform(0, 10) for _ in range(3)] for _ in range(n)]

def brute_force_skyline(points):
    skyline = []
    for p in points:
        dominated = False
        for q in points:
            if all(q[i] > p[i] for i in range(len(p))):
                dominated = True
        if not dominated:
            skyline.append(p)
    return skyline

def measure_runtime(n_values, repeat=3):
    runtimes = {}
   
    for n in n_values:
        times = []
        skyline_sizes = []
        for _ in range(repeat):
            points = generate_random_3d_points(n)
            start_time = time.time()
            skyline = compute_3d_skyline(points)
            end_time = time.time()
            run_time = end_time - start_time
            times.append(run_time)
            skyline_sizes.append(len(skyline))
        avg_time = sum(times) / len(times)
        avg_skyline_size = sum(skyline_sizes) / len(skyline_sizes)
        runtimes[n] = {
            'time': avg_time,
            'skyline_size': avg_skyline_size
        }
   
    return runtimes

def compute_theoretical_complexity(n, scale_factor=1e-7):
    if n <= 1:
        return 0
    return scale_factor * n * math.log(n)

def plot_runtime_vs_theoretical(n_values, runtimes, theoretical, brute_force, unoptimized, scale_factor):
    actual_times = [runtimes[n]['time'] for n in n_values]
    theoretical_times = [theoretical[n] for n in n_values]
    brute_force = [brute_force[n] for n in n_values]
    unoptimized = [unoptimized[n] for n in n_values]
    unopti_theoretical = [scale_factor * n * (math.log(n) ** 3) for n in n_values]
    n_2_theoretical = [scale_factor * n ** 2 for n in n_values]

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, actual_times, label="Part B runtime", marker='o')
    plt.plot(n_values, theoretical_times, label="O(n log n) Theoretical", marker='x')
    plt.plot(n_values, unoptimized, label="Part A runtime", marker='^')
    plt.plot(n_values, unopti_theoretical, label="O(n log^3 n) Theoretical", marker='d')
    plt.plot(n_values, brute_force, label="Brute Force Runtime", marker='s')
    plt.plot(n_values, n_2_theoretical, label="O(n^2) Theoretical", marker='v')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Dataset Size (n) (log scale)")
    plt.ylabel("Time (seconds) (log scale)")
    plt.title("Runtime vs Theoretical Runtime")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig('plot_part_b.png')

if __name__ == "__main__":
    n_values = [100, 500, 1000, 5000, 10000]
    runtimes = measure_runtime(n_values)
    smallest_n = n_values[0]
    scale_factor = runtimes[smallest_n]['time'] / (smallest_n * math.log(smallest_n))
    theoretical = {n: compute_theoretical_complexity(n, scale_factor) for n in n_values}
    brute_force_times ={}
    unoptimized_times = {}
    for n in n_values:
        points = generate_random_3d_points(n)
        st = time.time()
        skyline = brute_force_skyline(points)
        et = time.time()
        brute_force_time = et - st
        st = time.time()
        skyline = find_skyline(points)
        et = time.time()
        unoptimized = et - st
        brute_force_times[n] = brute_force_time
        unoptimized_times[n] = unoptimized

    plot_runtime_vs_theoretical(n_values, runtimes, theoretical, brute_force_times, unoptimized_times, scale_factor)

    # Sanity check
    test_points = [[np.random.uniform(0, 10) for _ in range(3)] for _ in range(10000)]
    correct_skyline = brute_force_skyline(test_points) # O(n^2) algo
    correct_skyline = set([tuple(item) for item in correct_skyline])
    skyline = compute_3d_skyline(test_points) # actual algo
    skyline = set([tuple(item) for item in skyline])
    assert skyline == correct_skyline, "Skyline results do not match!"
    print("Skyline results match!")