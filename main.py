import numpy as np
import matplotlib.pyplot as plt
import time
import math
from fixed_range_tree import find_skyline, brute_force_skyline, run_xd
from test_skyline import test_skyline_algorithm

def measure_runtime(n_values, d_values, repeat=3):
    runtimes = {}
   
    for d in d_values:
        runtimes[d] = {}
        for n in n_values:
            print(f"Testing n={n}, d={d}")
            times = []
            for _ in range(repeat):
                time_taken = run_xd(d, n)
                times.append(time_taken)
            avg_time = sum(times) / len(times)
            runtimes[d][n] = avg_time
   
    return runtimes

def compute_theoretical_complexity(n, d, scale_factor=1e-7):
    if n <= 1:
        return 0
    return scale_factor * n * (math.log(n) ** d)

def plot_performance():
    sample_sizes = [100, 500, 1000, 5000, 10000]
    dimensions = range(3, 7)
    
    timing = {}
    
    for dim in dimensions:
        timing[dim] = []
        for n in sample_sizes:
            print(f"Testing n={n}, d={dim}")
            timing[dim].append(run_xd(dim, n))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, dim in enumerate(dimensions):
        n_values = sample_sizes
        timing_values = timing[dim]
        
        # Find best scale factor for this dimension
        scale_factors = [1e-8, 1e-7, 1e-6]
        best_factor = scale_factors[0]
        best_error = float('inf')
        
        for factor in scale_factors:
            theoretical_values = [compute_theoretical_complexity(n, dim, factor) for n in n_values]
            error = sum((t1 - t2)**2 for t1, t2 in zip(timing_values, theoretical_values))
            if error < best_error:
                best_error = error
                best_factor = factor
        
        theoretical_values = [compute_theoretical_complexity(n, dim, best_factor) for n in n_values]

        ax = axes[idx]
        ax.loglog(n_values, timing_values, label="Measured Time", marker='o')
        ax.loglog(n_values, theoretical_values, label=f"Theoretical O(n·log^{dim}(n))", linestyle='--')
        ax.set_xlabel("Number of Points (n)")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(f"Timing for Dimension {dim}")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig('plot_part_a.png')
    plt.show()

if __name__ == "__main__":
    # test_skyline_algorithm()
    plot_performance()

    # test_points = [[np.random.uniform(0, 10) for _ in range(20)] for _ in range(100)]
    # correct_skyline = brute_force_skyline(test_points) # O(n^2) algo
    # correct_skyline = set([tuple(item) for item in correct_skyline])
    # skyline = find_skyline(test_points) # actual algo
    # skyline = set([tuple(item) for item in skyline])
    # assert skyline == correct_skyline, f"Skyline mismatch: {skyline} != {correct_skyline}"
    # print("Skyline results match!")