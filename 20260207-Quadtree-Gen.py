import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

# 1. Define the parameters
K1, K2 = 3, 5  # You can change these between 1 and 7
MAX_LEVEL = 8
MIN_LEVEL = 5  # Start with a coarse grid of 2^5 x 2^5

def target_function(x, y, k1, k2):
    return np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)

class QuadNode:
    def __init__(self, x, y, size, level):
        self.x = x
        self.y = y
        self.size = size
        self.level = level
        self.children = []
        self.val = None  # To store function value if leaf

    def random_subdivide(self, split_prob=0.4):
        """
        Recursively splits the node based on a random probability.
        Ignoring the function values entirely for the structure.
        """
        # Always split if we haven't hit the minimum level yet
        force_split = self.level < MIN_LEVEL
        
        # Stop splitting if we hit max level
        if self.level >= MAX_LEVEL:
            return

        # Roll the dice
        if force_split or random.random() < split_prob:
            half = self.size / 2
            # Create 4 children
            self.children = [
                QuadNode(self.x, self.y, half, self.level + 1),        # Bottom-Left
                QuadNode(self.x + half, self.y, half, self.level + 1), # Bottom-Right
                QuadNode(self.x, self.y + half, half, self.level + 1), # Top-Left
                QuadNode(self.x + half, self.y + half, half, self.level + 1) # Top-Right
            ]
            
            # Recurse
            for child in self.children:
                child.random_subdivide(split_prob)

    def collect_leaves(self, leaves_list):
        if not self.children:
            # It's a leaf, calculate function value at center
            center_x = self.x + self.size / 2
            center_y = self.y + self.size / 2
            self.val = target_function(center_x, center_y, K1, K2)
            leaves_list.append(self)
        else:
            for child in self.children:
                child.collect_leaves(leaves_list)

# 2. Build the Tree
root = QuadNode(0, 0, 1.0, 0) # Unit square domain [0,1]
root.random_subdivide(split_prob=0.45) # 45% chance to split at any given step after Level 3

# 3. Collect leaves for plotting
leaves = []
root.collect_leaves(leaves)

# 4. Visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.axis('off') # Hide axes for cleaner look

# Colormap
cmap = plt.get_cmap('viridis')

print(f"Total blocks generated: {len(leaves)}")

for node in leaves:
    # Normalize value from [-1, 1] to [0, 1] for the colormap
    normalized_color = (node.val + 1) / 2
    color = cmap(normalized_color)
    
    # Draw the rectangle
    rect = patches.Rectangle(
        (node.x, node.y), 
        node.size, 
        node.size, 
        linewidth=0.5, 
        edgecolor='black', 
        facecolor=color
    )
    ax.add_patch(rect)

plt.title(f"Random Quadtree (Level {MIN_LEVEL}-{MAX_LEVEL})\nFunction: sin(2$\\pi${K1}x)sin(2$\\pi${K2}y)", fontsize=14)
plt.savefig('quadtree_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'quadtree_plot.png'")