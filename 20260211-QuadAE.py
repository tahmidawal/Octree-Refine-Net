import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. Low-Level Operations: Morton Codes (2D)
# ==========================================

class Morton2D:
    """
    Handles 2D Morton Code (Z-Order Curve) encoding and decoding.
    Interleaves bits of X and Y.
    """
    @staticmethod
    def _interleave_bits(x):
        x = x & 0x0000FFFF
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        return x

    @staticmethod
    def _deinterleave_bits(x):
        x = x & 0x55555555
        x = (x | (x >> 1)) & 0x33333333
        x = (x | (x >> 2)) & 0x0F0F0F0F
        x = (x | (x >> 4)) & 0x00FF00FF
        x = (x | (x >> 8)) & 0x0000FFFF
        return x

    @staticmethod
    def xy2key(x, y, depth=16):
        kx = Morton2D._interleave_bits(x)
        ky = Morton2D._interleave_bits(y)
        key = kx | (ky << 1)
        return key

    @staticmethod
    def key2xy(key, depth=16):
        x = Morton2D._deinterleave_bits(key)
        y = Morton2D._deinterleave_bits(key >> 1)
        return x, y

# ==========================================
# 2. Quadtree Data Structure (O-CNN Style)
# ==========================================

class Quadtree:
    def __init__(self, max_depth, device='cpu'):
        self.max_depth = max_depth
        self.device = device
        
        # Storage per level
        self.keys = [None] * (max_depth + 1)
        self.children = [None] * (max_depth + 1)
        self.neighs = [None] * (max_depth + 1)
        self.features = [None] * (max_depth + 1)
        
        # Look-Up Tables for O(N) neighbor finding
        self._build_luts()

    def _build_luts(self):
        """Pre-computes parent-child neighbor relationships."""
        self.lut_parent = torch.zeros((4, 9), dtype=torch.long, device=self.device)
        self.lut_child = torch.zeros((4, 9), dtype=torch.long, device=self.device)
        
        for child_idx in range(4): # child (cy, cx)
            cy, cx = divmod(child_idx, 2) 
            for dir_idx in range(9):
                dy, dx = divmod(dir_idx, 3) 
                dy, dx = dy-1, dx-1 
                
                ny, nx = cy + dy, cx + dx
                
                py, px = 0, 0
                if ny < 0: py, ny = -1, ny + 2
                elif ny > 1: py, ny = 1, ny - 2
                if nx < 0: px, nx = -1, nx + 2
                elif nx > 1: px, nx = 1, nx - 2
                
                p_dir = (py + 1) * 3 + (px + 1)
                self.lut_parent[child_idx, dir_idx] = p_dir
                
                c_idx = ny * 2 + nx
                self.lut_child[child_idx, dir_idx] = c_idx

    def build_from_points(self, points, features):
        """
        points: (N, 2) normalized [0, 1]
        features: (N, C)
        """
        # 1. Quantize to finest grid
        res = 2 ** self.max_depth
        coords = (points * res).long().clamp(0, res - 1)
        
        # 2. Key Generation & Sorting (Finest Level)
        keys = Morton2D.xy2key(coords[:, 0], coords[:, 1])
        unique_keys, inv_idx = torch.unique(keys, sorted=True, return_inverse=True)
        
        self.keys[self.max_depth] = unique_keys.to(self.device)
        
        # Average features into unique keys
        feat_dim = features.shape[1]
        pooled_feat = torch.zeros((len(unique_keys), feat_dim), device=self.device)
        counts = torch.zeros((len(unique_keys), 1), device=self.device)
        
        # Note: .float() ensures types match for index_add_
        pooled_feat.index_add_(0, inv_idx.to(self.device), features.to(self.device).float())
        counts.index_add_(0, inv_idx.to(self.device), torch.ones_like(inv_idx, dtype=torch.float, device=self.device).unsqueeze(1))
        self.features[self.max_depth] = pooled_feat / counts.clamp(min=1)

        # 3. Build Hierarchy Bottom-Up
        for d in range(self.max_depth - 1, 0, -1):
            curr_keys = self.keys[d+1]
            parent_keys = curr_keys >> 2 # 2 bits per level
            
            unique_parents, inverse = torch.unique(parent_keys, sorted=True, return_inverse=True)
            self.keys[d] = unique_parents
            self.features[d] = torch.zeros((len(unique_parents), feat_dim), device=self.device)
            
            # Construct Children Pointers
            _, counts = torch.unique_consecutive(inverse, return_counts=True)
            starts = torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), torch.cumsum(counts, 0)[:-1]])
            self.children[d] = starts.int()

        # 4. Construct Neighbors
        for d in range(1, self.max_depth + 1):
            self._construct_neigh(d)

    def _construct_neigh(self, depth):
        if depth == 1:
            N = len(self.keys[depth])
            self.neighs[depth] = torch.full((N, 9), -1, dtype=torch.long, device=self.device)
            self._construct_neigh_geometric(depth)
            return

        curr_keys = self.keys[depth]
        parent_keys = curr_keys >> 2
        parent_idx = torch.searchsorted(self.keys[depth-1], parent_keys)
        parent_neighs = self.neighs[depth-1][parent_idx]
        
        child_type = curr_keys & 3 
        p_neigh_dir = self.lut_parent[child_type] 
        
        target_parent_idx = parent_neighs.gather(1, p_neigh_dir)
        
        mask = target_parent_idx != -1
        valid_parents = target_parent_idx[mask]
        
        child_start = torch.zeros_like(target_parent_idx)
        child_start[mask] = self.children[depth-1][valid_parents].long()
        
        target_child_offset = self.lut_child[child_type]
        
        # Verify existence
        x, y = Morton2D.key2xy(curr_keys, depth)
        dx = torch.tensor([-1, 0, 1], device=self.device).repeat_interleave(3)
        dy = torch.tensor([-1, 0, 1], device=self.device).repeat(3)
        
        nx = x.unsqueeze(1) + dx
        ny = y.unsqueeze(1) + dy
        expected_keys = Morton2D.xy2key(nx, ny, depth=depth)
        
        final_neighs = torch.searchsorted(self.keys[depth], expected_keys)
        final_neighs = final_neighs.clamp(0, len(self.keys[depth]) - 1)
        
        found = self.keys[depth][final_neighs] == expected_keys
        final_neighs[~found] = -1
        
        self.neighs[depth] = final_neighs

    def _construct_neigh_geometric(self, depth):
        keys = self.keys[depth]
        x, y = Morton2D.key2xy(keys, depth)
        offsets = torch.tensor([[-1,-1],[-1,0],[-1,1],[ 0,-1],[ 0,0],[ 0,1],[ 1,-1],[ 1,0],[ 1,1]], device=self.device)
        n_coords = torch.stack([x, y], dim=1).unsqueeze(1) + offsets.unsqueeze(0)
        n_keys = Morton2D.xy2key(n_coords[..., 0], n_coords[..., 1], depth=depth)
        idx = torch.searchsorted(keys, n_keys)
        idx = idx.clamp(0, len(keys) - 1)
        found = keys[idx] == n_keys
        idx[~found] = -1
        self.neighs[depth] = idx

# ==========================================
# 3. QuadConv Layers
# ==========================================

class QuadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.weights = nn.Linear(9 * in_channels, out_channels)
        
    def forward(self, features, quadtree, depth):
        neigh_idx = quadtree.neighs[depth]
        N, K = neigh_idx.shape
        
        pad_vec = torch.zeros((1, features.shape[1]), device=features.device)
        feat_padded = torch.cat([features, pad_vec], dim=0)
        
        gather_idx = neigh_idx.clone()
        gather_idx[gather_idx == -1] = N
        
        col = feat_padded[gather_idx]
        col_flat = col.view(N, -1)
        return self.weights(col_flat)

class QuadPool(nn.Module):
    def forward(self, features, quadtree, depth):
        keys = quadtree.keys[depth]
        parent_keys = keys >> 2
        parent_level_keys = quadtree.keys[depth-1]
        parent_idx = torch.searchsorted(parent_level_keys, parent_keys)
        
        pooled = torch.full((len(parent_level_keys), features.shape[1]), -1e9, device=features.device)
        # Use simple loop if index_reduce is unstable in older torch versions
        # pooled.index_reduce_(0, parent_idx, features, reduce='amax', include_self=True)
        for i in range(len(parent_idx)):
            p = parent_idx[i]
            pooled[p] = torch.maximum(pooled[p], features[i])
            
        pooled[pooled == -1e9] = 0
        return pooled, parent_idx

class QuadUnpool(nn.Module):
    def forward(self, features, quadtree, depth):
        keys = quadtree.keys[depth]
        parent_keys = keys >> 2
        parent_level_keys = quadtree.keys[depth-1]
        parent_idx = torch.searchsorted(parent_level_keys, parent_keys)
        return features[parent_idx]

class QuadAE(nn.Module):
    def __init__(self, in_c=1, base_c=16):
        super().__init__()
        # U-Net style architecture adapted for 8 levels
        self.enc1 = QuadConv(in_c, base_c)       # 8
        self.pool1 = QuadPool()
        self.enc2 = QuadConv(base_c, base_c*2)   # 7
        self.pool2 = QuadPool()
        self.enc3 = QuadConv(base_c*2, base_c*4) # 6
        
        self.unpool1 = QuadUnpool()
        self.dec1 = QuadConv(base_c*4, base_c*2) # 7
        self.unpool2 = QuadUnpool()
        self.dec2 = QuadConv(base_c*2, base_c)   # 8
        self.head = QuadConv(base_c, in_c)       # 8
        
    def forward(self, qt, max_depth):
        d8 = qt.features[max_depth]
        
        # Encoder
        x8 = F.relu(self.enc1(d8, qt, max_depth))
        x7_p, _ = self.pool1(x8, qt, max_depth)
        
        x7 = F.relu(self.enc2(x7_p, qt, max_depth-1))
        x6_p, _ = self.pool2(x7, qt, max_depth-1)
        
        # Bottleneck
        x6 = F.relu(self.enc3(x6_p, qt, max_depth-2))
        
        # Decoder
        x7_up = self.unpool1(x6, qt, max_depth-1)
        x7_dec = F.relu(self.dec1(x7_up, qt, max_depth-1))
        
        x8_up = self.unpool2(x7_dec, qt, max_depth)
        x8_dec = F.relu(self.dec2(x8_up, qt, max_depth))
        
        out = self.head(x8_dec, qt, max_depth)
        return out

# ==========================================
# 4. User's Data Logic (Integrated)
# ==========================================

K1, K2 = 3, 5  
MAX_LEVEL = 8
MIN_LEVEL = 5  

def target_function(x, y, k1, k2):
    return np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)

class QuadNode:
    def __init__(self, x, y, size, level):
        self.x, self.y, self.size, self.level = x, y, size, level
        self.children, self.val = [], None 

    def random_subdivide(self, split_prob=0.4):
        force_split = self.level < MIN_LEVEL
        if self.level >= MAX_LEVEL: return

        if force_split or random.random() < split_prob:
            half = self.size / 2
            self.children = [
                QuadNode(self.x, self.y, half, self.level + 1),
                QuadNode(self.x + half, self.y, half, self.level + 1),
                QuadNode(self.x, self.y + half, half, self.level + 1),
                QuadNode(self.x + half, self.y + half, half, self.level + 1)
            ]
            for child in self.children:
                child.random_subdivide(split_prob)

    def collect_leaves(self, leaves_list):
        if not self.children:
            center_x = self.x + self.size / 2
            center_y = self.y + self.size / 2
            self.val = target_function(center_x, center_y, K1, K2)
            leaves_list.append(self)
        else:
            for child in self.children:
                child.collect_leaves(leaves_list)

def generate_user_data():
    """Generates data using the user's original recursive logic."""
    root = QuadNode(0, 0, 1.0, 0)
    root.random_subdivide(split_prob=0.45)
    
    leaves = []
    root.collect_leaves(leaves)
    
    points = []
    values = []
    
    # Rasterization Step: Convert variable sized leaves to finest grid points
    step = 1.0 / (2**MAX_LEVEL)
    
    for node in leaves:
        # Determine how many fine pixels this node covers
        # Node size is 1 / 2^level. Fine pixel is 1 / 2^MAX_LEVEL
        # Scale factor = 2^(MAX - level)
        scale = int(2**(MAX_LEVEL - node.level))
        
        start_x = node.x + step/2
        start_y = node.y + step/2
        
        # Fill the area covered by this leaf
        for i in range(scale):
            for j in range(scale):
                px = start_x + i*step
                py = start_y + j*step
                points.append([px, py])
                values.append([node.val])
                
    # Cast to float32 to match network weights
    return torch.tensor(points, dtype=torch.float32), torch.tensor(values, dtype=torch.float32)

# ==========================================
# 5. Main Execution
# ==========================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    # 1. Generate Data (User Logic)
    print("Generating Adaptive Data from QuadNodes...")
    pts, feats = generate_user_data()
    print(f"Generated {len(pts)} fine-level pixels from adaptive mesh.")
    
    # 2. Build Quadtree
    print("Building O-CNN Quadtree...")
    qt = Quadtree(max_depth=MAX_LEVEL, device=device)
    qt.build_from_points(pts, feats)
    print(f"Quadtree built. Nodes per level: {[len(k) for k in qt.keys if k is not None]}")
    
    # 3. Model
    model = QuadAE(in_c=1, base_c=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 4. Training
    print("Training...")
    losses = []
    target = qt.features[MAX_LEVEL]
    
    for epoch in range(101):
        optimizer.zero_grad()
        recon = model(qt, MAX_LEVEL)
        loss = criterion(recon, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")
            
    # 5. Visualization
    print("Visualizing...")
    keys = qt.keys[MAX_LEVEL].cpu()
    x, y = Morton2D.key2xy(keys)
    res = 2**MAX_LEVEL
    
    x_np = x.numpy() / res
    y_np = y.numpy() / res
    val_gt = target.detach().cpu().numpy().flatten()
    val_pred = recon.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth (Adaptive->Fine)")
    plt.scatter(x_np, y_np, c=val_gt, cmap='viridis', s=1)
    plt.axis('equal')
    
    plt.subplot(1, 3, 2)
    plt.title("O-CNN Reconstruction")
    plt.scatter(x_np, y_np, c=val_pred, cmap='viridis', s=1)
    plt.axis('equal')
    
    plt.subplot(1, 3, 3)
    plt.title("Training Loss")
    plt.plot(losses)
    
    plt.tight_layout()
    plt.savefig('quadnet_user_result.png')
    print("Saved to quadnet_user_result.png")

if __name__ == "__main__":
    main()