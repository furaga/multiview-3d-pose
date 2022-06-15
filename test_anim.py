from venv import create
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import lib.visualize as visualize

# Fixing random state for reproducibility
np.random.seed(19680801)


def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk


def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    return lines


ax = visualize.create_plt()
visualize.show_plt(False)

for i in range(100):
    visualize.anim_begin_update(ax)
    p = np.random.random(3) - 0.5
    ax.scatter([p[0]], [p[1]], [p[2]], c=(1, 0, 0), s=100)
    visualize.anim_end_update(ax)
