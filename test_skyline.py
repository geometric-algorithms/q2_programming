import numpy as np
from fixed_range_tree import find_skyline, brute_force_skyline, generate_random_points
from skyline_3d import compute_3d_skyline

def test_skyline_algorithm():
    print("RUNNING COMPREHENSIVE SKYLINE ALGORITHM TESTS")
    
    # Test 1: Simple 2D test
    test_2d = [[1, 2], [3, 4], [2, 3], [5, 1], [4, 2]]
    correct_2d = brute_force_skyline(test_2d)
    skyline_2d = find_skyline(test_2d)
    assert set(map(tuple, skyline_2d)) == set(map(tuple, correct_2d)), f"2D test failed: {skyline_2d} != {correct_2d}"
    print("✓ 2D test passed")

    test = [[np.random.uniform(0, 10) for _ in range(3)] for _ in range(10)]
    correct = brute_force_skyline(test)
    skyline = compute_3d_skyline(test)
    assert set(map(tuple, skyline)) == set(map(tuple, correct)), f"3D test failed: {skyline} != {correct}"
    print(skyline, "skyline found")
    print("✓ 3D test passed")
    
    # Test 2: Edge cases
    edge_cases = [
        # Empty set
        [],
        # Single point
        [[1, 2, 3]],
        # Duplicate points
        [[1, 2, 3], [1, 2, 3], [4, 5, 6]],
        # Points with some identical coordinates
        [[1, 2, 3], [1, 2, 4], [1, 3, 2]],
        # All points dominated by one point
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    ]
    
    for i, test in enumerate(edge_cases):
        correct = brute_force_skyline(test)
        skyline = find_skyline(test)
        assert set(map(tuple, skyline)) == set(map(tuple, correct)), f"Edge case {i}: {skyline} != {correct}"
    print("✓ Edge cases passed")
    
    # Test 3: Stress test with random points
    for dim in range(2, 15):
        for n in [10, 50, 100]:
            np.random.seed(42 + dim + n)  # For reproducibility
            points = generate_random_points(n, dim)
            correct = brute_force_skyline(points)
            skyline = find_skyline(points)
            assert set(map(tuple, skyline)) == set(map(tuple, correct)), \
                f"Random test d={dim}, n={n} failed"
    print(f"✓ Random tests passed (dimensions 2-5, sizes 10-100)")
    
    # Test 4: Corner case with leaf nodes
    for _ in range(5):
        n = np.random.randint(5, 15)
        d = np.random.randint(2, 5)
        np.random.seed(42)  # Reproducible tests
        points = [[np.random.randint(0, 10) for _ in range(d)] for _ in range(n)]
        correct_skyline = brute_force_skyline(points)
        skyline = find_skyline(points)
        correct_set = set(tuple(item) for item in correct_skyline)
        skyline_set = set(tuple(item) for item in skyline)
        assert skyline_set == correct_set, \
            f"Corner case failed: {skyline_set} != {correct_set}"
    print("✓ Corner cases with leaf nodes passed")
    
    # Test 5: Degenerate cases (aligned points)
    aligned_cases = [
        # All points on same x-coordinate
        [(5, i) for i in range(10)],
        # All points on same y-coordinate
        [(i, 5) for i in range(10)],
        # Grid points with duplicates
        [(i % 3, j % 4) for i in range(10) for j in range(10)]
    ]
    
    for i, test in enumerate(aligned_cases):
        correct = brute_force_skyline(test)
        skyline = find_skyline(test)
        assert set(map(tuple, skyline)) == set(map(tuple, correct)), \
            f"Aligned case {i}: {skyline} != {correct}"
    print("✓ Degenerate aligned points cases passed")
    
    print("ALL TESTS PASSED! The implementation is correct.")

if __name__ == "__main__":
    test_skyline_algorithm()