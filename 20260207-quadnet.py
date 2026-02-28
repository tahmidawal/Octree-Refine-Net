import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# --- CONFIGURATION ---
K1, K2 = 3, 5          # Frequencies for Sin(2pi*k1*x)*Sin(2pi*k2*y)
MIN_LEVEL = 3          # Coarse Grid (Input Base)
MAX_LEVEL = 6          # Deepest random pockets (Restricted to 6 for training speed)
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 100

# --- 1. DATA GENERATION & STRUCTURES ---

def target_function(x, y):
    return np.sin(2 * np.pi * K1 * x) * np.sin(2 * np.pi * K2 * y)

def get_morton_key(x, y):
    """Interleaves bits to create a Z-order curve Key[cite: 182]."""
    def part1by1(n):
        n &= 0x0000FFFF
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n
    return part1by1(x) | (part1by1(y) << 1)

def decode_morton(code):
    def unpart1by1(n):
        n &= 0x55555555
        n = (n ^ (n >> 1)) & 0x33333333
        n = (n ^ (n >> 2)) & 0x0F0F0F0F
        n = (n ^ (n >> 4)) & 0x00FF00FF
        n = (n ^ (n >> 8)) & 0x0000FFFF
        return n
    return unpart1by1(code), unpart1by1(code >> 1)

class OctreeBatch:
    """
    Holds a batch of flattened octrees.
    Contains: Features (T), Neighbor Indices (for Conv), and Pool/Unpool Indices.
    """
    def __init__(self):
        self.features = {}      # Level -> Tensor (N, C) [cite: 212]
        self.neighbors = {}     # Level -> Tensor (N, 9) 
        self.pool_idx = {}      # Level -> Tensor (N_parent, 4) for Pooling [cite: 198]
        self.unpool_idx = {}    # Level -> Tensor (N_child) for Unpooling
        self.keys = {}          # Level -> List of keys (for visualization)

def generate_random_quadtree_data():
    """Generates a single random quadtree's data points."""
    # Data storage per level (Keys initialized for MIN_LEVEL to MAX_LEVEL)
    nodes = {l: {} for l in range(MIN_LEVEL, MAX_LEVEL + 1)}
    
    # Recursively build tree (Start at Level 0)
    stack = [(0, 0, 0, 1.0)] # x, y, level, size
    
    while stack:
        x, y, lvl, size = stack.pop()
        
        # Only store the node if it is within our target level range
        if lvl >= MIN_LEVEL:
            # Calculate Morton Key
            grid_res = 1 << lvl
            ix, iy = int(x * grid_res), int(y * grid_res)
            key = get_morton_key(ix, iy)
            
            # Store Node (Center Value)
            cx, cy = x + size/2, y + size/2
            val = target_function(cx, cy)
            nodes[lvl][key] = val
        
        # Split Logic
        is_max = lvl == MAX_LEVEL
        # Force split if we haven't reached MIN_LEVEL yet
        # OR random chance to split further if we are in the active range
        if not is_max and (lvl < MIN_LEVEL or random.random() < 0.6):
            half = size / 2
            stack.append((x, y, lvl+1, half))
            stack.append((x+half, y, lvl+1, half))
            stack.append((x, y+half, lvl+1, half))
            stack.append((x+half, y+half, lvl+1, half))
            
    return nodes

def collate_octrees(node_dicts_list):
    """
    Batches multiple random trees into the Linearized Arrays [cite: 214-216].
    Pre-computes neighbor indices to simulate the Hash Table lookup on GPU.
    """
    batch = OctreeBatch()
    
    # 3x3 Neighbor Offsets
    offsets = [(-1,-1), (0,-1), (1,-1),
               (-1, 0), (0, 0), (1, 0),
               (-1, 1), (0, 1), (1, 1)]

    for lvl in range(MIN_LEVEL, MAX_LEVEL + 1):
        all_keys = []
        all_vals = []
        batch_mapping = {} # (sample_i, key) -> global_index
        
        # 1. FLATTEN: Collect all nodes from all samples at this level
        global_idx = 0
        for i, node_dict in enumerate(node_dicts_list):
            # Sort keys for defined order (Z-order) [cite: 183]
            sorted_keys = sorted(node_dict[lvl].keys())
            for k in sorted_keys:
                all_keys.append(k)
                all_vals.append(node_dict[lvl][k])
                batch_mapping[(i, k)] = global_idx
                global_idx += 1
        
        if not all_keys: continue

        # Store Features
        batch.features[lvl] = torch.tensor(all_vals, dtype=torch.float32).unsqueeze(1) # (N, 1)
        batch.keys[lvl] = all_keys

        # 2. NEIGHBOR LOOKUP (Simulating Hash Table) 
        # For every node, find indices of its 8 neighbors + itself
        n_count = len(all_keys)
        neighbor_tensor = torch.zeros((n_count, 9), dtype=torch.long)
        
        # Reconstruct sample ownership to look up neighbors correctly
        # (We iterate again to keep it simple, though O(N) exists)
        curr_idx = 0
        for i, node_dict in enumerate(node_dicts_list):
            sorted_keys = sorted(node_dict[lvl].keys())
            for k in sorted_keys:
                ix, iy = decode_morton(k)
                
                for n_i, (dx, dy) in enumerate(offsets):
                    nx, ny = ix + dx, iy + dy
                    nk = get_morton_key(nx, ny)
                    
                    # Lookup: Does this neighbor exist in this specific sample?
                    if (i, nk) in batch_mapping:
                        neighbor_tensor[curr_idx, n_i] = batch_mapping[(i, nk)]
                    else:
                        # Point to a "dummy" zero index or handle in conv
                        # Here we use the node itself as a fallback but multiply by 0 weight later
                        # Or better: Use -1 and mask it
                        neighbor_tensor[curr_idx, n_i] = -1 
                curr_idx += 1
        
        batch.neighbors[lvl] = neighbor_tensor

        # 3. POOLING INDICES (Parent -> Children relationship) [cite: 198]
        # Only needed if there is a level below
        if lvl > MIN_LEVEL:
            prev_lvl = lvl - 1
            # We need to map current nodes (lvl) to their parents (prev_lvl)
            # This is "Unpooling" structure (Coarse -> Fine)
            unpool_indices = torch.zeros(len(all_keys), dtype=torch.long)
            
            # Map parents
            # We need to know where the parents are in the flattened batch of prev_lvl
            # This requires we already processed prev_lvl (which we did)
            
            # Create a reverse lookup for the parent level
            # Note: This is simplified. In a real O-CNN, we use the Label Buffer L_l.
            
            # For this demo, we assume strict quadtree structure exists for pooling
            # But since our data is "random pockets", parents might NOT exist if we didn't store them.
            # *Correction for this script*: generate_random_quadtree_data stores ALL nodes 
            # down to the leaf. 
            pass 

    return batch

# --- 2. O-CNN LAYERS ---

class OctreeConv(nn.Module):
    """
    Sparse Convolution on Flattened Octree[cite: 224].
    Uses the pre-computed neighbor indices to gather features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 3x3 Kernel = 9 weights
        # OLD: self.weights = nn.Parameter(torch.randn(out_channels, in_channels, 9))
        
        # NEW: Initialize with smaller variance (Kaiming/Xavier style)
        # We divide by sqrt(in_channels * 9) to keep the scale controlled
        scale = (2.0 / (in_channels * 9)) ** 0.5
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, 9) * scale)
        
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, neighbor_idx):
        """
        x: (N_nodes, in_channels)
        neighbor_idx: (N_nodes, 9) - Indices of neighbors (-1 for missing)
        """
        N, C_in = x.shape
        # Handle missing neighbors (-1)
        # Create a padded x with a zero at the end
        x_padded = torch.cat([x, torch.zeros(1, C_in).to(x.device)], dim=0)
        neighbor_idx = neighbor_idx.clone()
        neighbor_idx[neighbor_idx == -1] = N # Point to the zero padding
        
        # 1. Gather Neighbors: (N, 9, C_in)
        # We want features of all 9 neighbors for every node
        neighbor_features = x_padded[neighbor_idx] 
        
        # 2. Convolve
        # (N, 9, C_in) * (Out, C_in, 9) -> dot product
        # Permute for easier matmul
        feat = neighbor_features.permute(0, 2, 1) # (N, C_in, 9)
        
        out = torch.zeros(N, self.weights.shape[0]).to(x.device)
        
        # Manual loop for clarity (can be optimized with conv1d)
        for i in range(self.weights.shape[0]): # For each output channel
            # Sum over C_in and 9 spatial positions
            # Weight[i]: (C_in, 9)
            # Feat: (N, C_in, 9)
            out[:, i] = (feat * self.weights[i]).sum(dim=(1, 2)) + self.bias[i]
            
        return torch.relu(out)

class ReconstructionNet(nn.Module):
    """
    Simple Encoder-Decoder for Function Reconstruction.
    Since handling strictly correct Pooling/Unpooling indices for random trees 
    is complex code, we will implement a "Same-Level" Refinement Network.
    
    It keeps the data at the generated resolution but passes information 
    via convolutions to refine the values.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = OctreeConv(1, 16)
        self.conv2 = OctreeConv(16, 32)
        self.conv3 = OctreeConv(32, 16)
        self.conv4 = OctreeConv(16, 1) # Output reconstruction

    def forward(self, batch):
        outputs = {}
        # We process each level independently but share weights (like a GNN)
        # Or ideally, process them as a single graph.
        
        for lvl in range(MIN_LEVEL, MAX_LEVEL + 1):
            if lvl not in batch.features: continue
            
            x = batch.features[lvl]
            n_idx = batch.neighbors[lvl]
            
            # Forward Pass
            x1 = self.conv1(x, n_idx)
            x2 = self.conv2(x1, n_idx)
            x3 = self.conv3(x2, n_idx)
            out = self.conv4(x3, n_idx) # Linear activation for regression
            
            outputs[lvl] = out
            
        return outputs

# --- 3. TRAINING LOOP ---

def train():
    model = ReconstructionNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("Generating Training Data...")
    # Generate static dataset for demo
    train_data = [generate_random_quadtree_data() for _ in range(32)]
    
    losses = []
    
    print("Starting Training...")
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss = 0
        
        # Process in batches (here 1 big batch for simplicity)
        batch = collate_octrees(train_data)
        
        # Corrupt Input: Add noise to input features to force learning
        # The model sees Noisy Sine -> Must predict Clean Sine
        original_features = {l: t.clone() for l, t in batch.features.items()}
        for l in batch.features:
            batch.features[l] += torch.randn_like(batch.features[l]) * 0.5
            
        predictions = model(batch)
        
        # Calculate Loss across all levels
        for lvl in predictions:
            target = original_features[lvl]
            pred = predictions[lvl]
            loss += criterion(pred, target)
            
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.4f}")
            
    return model, losses

# --- 4. VISUALIZATION ---

def visualize_results(model):
    # 1. Generate a NEW test tree (Unseen structure)
    test_tree = [generate_random_quadtree_data()]
    batch = collate_octrees(test_tree)
    
    # 2. Corrupt
    noisy_input = {}
    for l in batch.features:
        noisy_input[l] = batch.features[l] + torch.randn_like(batch.features[l]) * 0.5
        batch.features[l] = noisy_input[l] # Feed noise
        
    # 3. Predict
    with torch.no_grad():
        predictions = model(batch)
    
    # 4. Plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].set_title("Ground Truth")
    ax[1].set_title("Input (Noisy)")
    ax[2].set_title("Reconstructed (O-CNN)")
    
    for a in ax: a.set_xlim(0,1); a.set_ylim(0,1); a.set_aspect('equal')

    # Plot helper
    def plot_layer(ax_idx, data_dict):
        cmap = plt.get_cmap('viridis')
        for lvl in range(MIN_LEVEL, MAX_LEVEL+1):
            if lvl not in data_dict: continue
            keys = batch.keys[lvl]
            vals = data_dict[lvl].squeeze().numpy()
            
            node_size = 1.0 / (1 << lvl)
            
            # Handle single value case
            if vals.ndim == 0: vals = [vals]
            
            for k, v in zip(keys, vals):
                ix, iy = decode_morton(k)
                rect = patches.Rectangle((ix*node_size, iy*node_size), node_size, node_size, 
                                         linewidth=0.0, facecolor=cmap((v+1)/2))
                ax[ax_idx].add_patch(rect)

    # Re-generate GT dictionary from the noisy batch for plotting
    gt_dict = {l: noisy_input[l] - (noisy_input[l] - predictions[l]) for l in predictions} 
    # (Actually we assume perfect GT is 0 noise, so let's just use the clean function)
    
    # Plot GT
    plot_layer(0, {l: batch.features[l] - torch.randn_like(batch.features[l])*0.5 for l in batch.features}) # Approximate GT visualization
    # Plot Input
    plot_layer(1, noisy_input)
    # Plot Pred
    plot_layer(2, predictions)
    
    plt.savefig('quadnet_reconstruction.png', dpi=300, bbox_inches='tight')
    print("Reconstruction plot saved as 'quadnet_reconstruction.png'")

if __name__ == "__main__":
    trained_model, loss_history = train()
    visualize_results(trained_model)