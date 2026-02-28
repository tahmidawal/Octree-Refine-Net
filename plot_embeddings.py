import numpy as np
import matplotlib.pyplot as plt
import re

def parse_and_plot_embeddings(filepath, title):
    with open(filepath, 'r') as file:
        data = file.read()
    
    # Extract Depth 0 and Depth 1 L2 norms as a quick visualization
    l2_norms = [float(x) for x in re.findall(r'L2=([0-9.]+)', data)]
    depth_counts = [1, 4, 16, 64, 256, 1024, 4096, 5888] # From your dump
    
    avg_l2_per_depth = []
    idx = 0
    for count in depth_counts:
        if idx + count <= len(l2_norms):
            avg_l2_per_depth.append(np.mean(l2_norms[idx:idx+count]))
            idx += count
            
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(avg_l2_per_depth)), avg_l2_per_depth, marker='o', linewidth=2)
    plt.title(f'Average L2 Norm of Embeddings vs. Depth ({title})')
    plt.xlabel('Quadtree Depth')
    plt.ylabel('Average L2 Norm')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    filename = f'plots/embeddings_{title.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.show()

# Run for both files
parse_and_plot_embeddings('plots/val_poisson_20260219_045144/sample_03_k2_5_emb_f.txt', 'Source f')
parse_and_plot_embeddings('plots/val_poisson_20260219_045144/sample_03_k2_5_emb_u.txt', 'Solution u')
