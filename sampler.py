import numpy as np


sample_space = {
    "stem_width": np.arange(16, 128 + 8, step=8),
    "width_multiplier": [1.5, 2, 2.5, 3, 3.5],
    "stage_depth": (2, 16),
    "bottleneck_ratio": [1/4, 1/8, 1/16, 1/32],
    "groups": [16, 32, 64],
    "att_ratio": [1/16, 1/8, 1/4],
    "graph_bottleneck_ratio": [1, 1/2, 1/4]
}


def sample(n_stage=4):
    stem_width = np.random.choice(sample_space["stem_width"])

    min_depth, max_depth = sample_space["stage_depth"]
    stages_dept = [np.random.randint(min_depth, max_depth) for _ in range(n_stage)]
    depth = sum(stages_dept)

    u = np.zeros(depth)
    for i in range(depth):
        u[i] = stem_width * (i + 1)

    width_mul = np.random.choice(sample_space["width_multiplier"])
    s = np.log(u / stem_width) / np.log(width_mul)
    s = np.ceil(s)
    w = np.round(stem_width * width_mul ** s).astype(int)

    ws = []
    start = 0
    for i in range(n_stage):
        ws.append(w[start:start + stages_dept[i]])
        start += stages_dept[i]

    return {
        "stem_width": stem_width,
        "stages_width": ws,
        "bottleneck_ratio": np.random.choice(sample_space["bottleneck_ratio"]),
        "groups": np.random.choice(sample_space["groups"]),
        "att_ratio": np.random.choice(sample_space["att_ratio"]),
        "graph_bottleneck_ratio": np.random.choice(sample_space["graph_bottleneck_ratio"])
    }
