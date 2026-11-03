import numpy as np
from india_soc_fire.cv.spatial_blocks import hex_block_ids, hex_block_folds

def test_nearby_same_block_far_different():
    ids = hex_block_ids([0, 100, 200_000], [0, 100, 0], 25)
    assert ids[0] == ids[1] and ids[0] != ids[2]

def test_block_size_scales():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 300_000, 2000), rng.uniform(0, 300_000, 2000)
    assert len(np.unique(hex_block_ids(x, y, 25))) > len(np.unique(hex_block_ids(x, y, 100)))

def test_points_within_radius():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 200_000, 3000), rng.uniform(0, 200_000, 3000)
    ids = hex_block_ids(x, y, 50)
    R = 50_000 / np.sqrt(3)
    for b in np.unique(ids):
        m = ids == b
        cx, cy = x[m].mean(), y[m].mean()
        assert np.max(np.hypot(x[m]-cx, y[m]-cy)) <= 2 * R + 1

def test_folds_keep_blocks_whole_and_balanced():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 400_000, 4000), rng.uniform(0, 400_000, 4000)
    ids = hex_block_ids(x, y, 25); folds = hex_block_folds(x, y, 25, 5)
    for b in np.unique(ids):
        assert len(np.unique(folds[ids == b])) == 1
    counts = np.bincount(folds)
    assert len(counts) == 5 and counts.min() > 0.8 * counts.mean()
