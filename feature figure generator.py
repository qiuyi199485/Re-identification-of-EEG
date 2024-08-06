import matplotlib.pyplot as plt
import numpy as np

# Define dark colors
dark_colors = ['#2F4F4F', '#696969', '#708090', '#778899', '#808080', '#A9A9A9', '#000000', '#1C1C1C', '#2E2E2E', '#4C4C4C']

# Generate Random Color Rectangle Chart
def generate_random_feature_map(rows, cols):
    data = np.random.choice(len(dark_colors), (rows, cols))
    plt.figure(figsize=(10, 5))
    
    
    plt.gca().add_patch(plt.Rectangle((0, 0), cols, rows, fill=None, edgecolor='darkred', linewidth=2))
    
    for i in range(rows):
        for j in range(cols):
            color_idx = data[i, j]
            plt.gca().add_patch(plt.Rectangle((j, rows - i - 1), 1, 1, color=dark_colors[color_idx]))

    plt.xlim(0, cols)
    plt.ylim(0, rows)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()

# Generate 10x5 feature maps
generate_random_feature_map(10, 5)
