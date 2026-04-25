"""
poisson_vitsep_linearcp.py
--------------------------
ViT encoder + Linear CP-tensor decoder for 3D Poisson NM-ROM.

Key insight: the SeparableDecoder's deep MLP creates a highly nonlinear z→h
mapping that makes GN unreliable. The ViT decoder works because it's near-linear.
This version makes h a near-linear function of z while keeping CP structure:

    h = W_rank @ (z + gelu(W_hidden @ z))   # mild nonlinearity with skip
    u[i,j,k] = Σ_r h[r] * W_x[r,i] * W_y[r,j] * W_z[r,k]

The skip connection ensures the Jacobian ∂u/∂z is dominated by the linear
term W_rank, making GN convergence reliable from cold-start (z=0).
Still EQ-compatible: evaluate at any node in O(rank).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.scipy.sparse.linalg as jax_linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Sequence

USE_CHECKPOINT  = '--use-checkpoint'  in sys.argv
SKIP_CHECKPOINT = '--skip-checkpoint' in sys.argv

# ==========================================
# 0. Logging
# ==========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
(SCRIPT_DIR / 'plots').mkdir(parents=True, exist_ok=True)

LOG_FILE = SCRIPT_DIR / 'vitsep_linearcp_training.log'
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = TeeLogger(LOG_FILE)

# ==========================================
# 1. Domain Setup
# ==========================================
N         = 32
num_nodes = N ** 3
k_dim     = 12
L         = 1.0
dx        = L / (N - 1)

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

def K_op_3d(u_flat):
    u   = u_flat.reshape((N, N, N))
    out = jnp.zeros_like(u)
    out = out.at[1:-1, 1:-1, 1:-1].set(
        (6*u[1:-1,1:-1,1:-1]
         - u[0:-2,1:-1,1:-1] - u[2:,1:-1,1:-1]
         - u[1:-1,0:-2,1:-1] - u[1:-1,2:,1:-1]
         - u[1:-1,1:-1,0:-2] - u[1:-1,1:-1,2:]) / dx**2
    )
    out = out.at[0,:,:].set(u[0,:,:])
    out = out.at[-1,:,:].set(u[-1,:,:])
    out = out.at[:,0,:].set(u[:,0,:])
    out = out.at[:,-1,:].set(u[:,-1,:])
    out = out.at[:,:,0].set(u[:,:,0])
    out = out.at[:,:,-1].set(u[:,:,-1])
    return out.flatten()

def get_F_3d(k1, k2, k3):
    F = jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z) * 10.0
    F = F.at[0,:,:].set(0.).at[-1,:,:].set(0.)
    F = F.at[:,0,:].set(0.).at[:,-1,:].set(0.)
    F = F.at[:,:,0].set(0.).at[:,:,-1].set(0.)
    return F.flatten()

def get_analytical_3d(k1, k2, k3):
    c = 10.0 / ((k1**2 + k2**2 + k3**2) * jnp.pi**2)
    return (c * jnp.sin(k1*jnp.pi*X)
              * jnp.sin(k2*jnp.pi*Y)
              * jnp.sin(k3*jnp.pi*Z)).flatten()

mask_3d = jnp.ones((N, N, N))
mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
mask    = mask_3d.flatten()
u_g     = jnp.zeros(num_nodes)

# ==========================================
# 2. Model Definitions
# ==========================================

class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(h, h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(int(self.embed_dim * self.mlp_ratio))(h)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        x = x + h
        return x


class LinearCPDecoder(nn.Module):
    """
    Shallow CP-tensor decoder. Uses a 2-layer MLP with skip connection
    to map z→h, keeping the mapping mild enough for GN while being
    expressive enough to learn diverse solution shapes.

    h = W_rank @ swish(W2 @ swish(W1 @ z)) + W_direct @ z
        ^--- nonlinear path (expressive) ---^  ^-- linear skip (GN anchor) --^

    u[i,j,k] = Σ_r h[r] * W_x[r,i] * W_y[r,j] * W_z[r,k] + bias

    EQ-compatible: evaluate at any (i,j,k) in O(rank).
    """
    latent_dim:  int
    rank:        int = 512
    grid_size:   int = 32
    hidden_dim:  int = 256

    def setup(self):
        self.W1       = nn.Dense(self.hidden_dim)
        self.W2       = nn.Dense(self.hidden_dim)
        self.W_rank   = nn.Dense(self.rank)
        self.W_direct = nn.Dense(self.rank)   # linear skip: z → rank directly

        init = nn.initializers.normal(0.01)
        Ng   = self.grid_size
        self.W_x  = self.param('W_x',  init, (self.rank, Ng))
        self.W_y  = self.param('W_y',  init, (self.rank, Ng))
        self.W_z  = self.param('W_z',  init, (self.rank, Ng))
        self.bias = self.param('bias', nn.initializers.zeros, ())

    def __call__(self, z):
        # Nonlinear path: 2 hidden layers
        h_nonlin = nn.swish(self.W1(z))
        h_nonlin = nn.swish(self.W2(h_nonlin))
        h_nonlin = self.W_rank(h_nonlin)       # (rank,)

        # Linear skip: direct projection z → rank
        h_linear = self.W_direct(z)            # (rank,)

        # Sum: linear baseline + nonlinear correction
        h = h_linear + h_nonlin

        u_3d = jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z)
        return u_3d.flatten() + self.bias


class ViTLinearCPAutoencoder(nn.Module):
    """ViT encoder (mean pool) + LinearCPDecoder."""
    latent_dim:     int
    num_nodes:      int
    patch_size:     int   = 8
    embed_dim:      int   = 64
    num_heads:      int   = 4
    num_enc_layers: int   = 4
    rank:           int   = 512
    hidden_dim:     int   = 64

    def setup(self):
        grid_n         = round(self.num_nodes ** (1/3))
        n_per_side     = grid_n // self.patch_size
        patch_dim      = self.patch_size ** 3

        self._grid_n      = grid_n
        self._n_per_side  = n_per_side
        self._num_patches = n_per_side ** 3
        self._patch_dim   = patch_dim

        self.patch_embed = nn.Dense(self.embed_dim)
        self.enc_pos     = self.param(
            'enc_pos',
            nn.initializers.normal(stddev=0.02),
            (self._num_patches, self.embed_dim)
        )
        self.enc_blocks = [
            TransformerBlock(self.embed_dim, self.num_heads)
            for _ in range(self.num_enc_layers)
        ]
        self.enc_norm = nn.LayerNorm()
        self.enc_proj = nn.Dense(self.latent_dim)

        self.decoder = LinearCPDecoder(
            latent_dim=self.latent_dim,
            rank=self.rank,
            grid_size=self._grid_n,
            hidden_dim=self.hidden_dim,
        )

    def _patchify(self, u_flat):
        n = self._n_per_side
        p = self.patch_size
        return (u_flat
                .reshape(n, p, n, p, n, p)
                .transpose(0, 2, 4, 1, 3, 5)
                .reshape(self._num_patches, self._patch_dim))

    def encode(self, u):
        x = self._patchify(u)
        x = self.patch_embed(x)
        x = x + self.enc_pos
        for block in self.enc_blocks:
            x = block(x)
        x = self.enc_norm(x)
        z = x.mean(axis=0)
        z = self.enc_proj(z)
        return z

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, u):
        return self.decode(self.encode(u))


# ==========================================
# 3. Training Data
# ==========================================
def full_order_fem_solver(F_vec, u_guess=None, tol=1e-6):
    if u_guess is None:
        u_guess = jnp.zeros(num_nodes)
    u, _ = jax_linalg.cg(K_op_3d, F_vec, x0=u_guess, tol=tol)
    return u

print(f"JAX devices: {jax.devices()}")
print(f"Grid: {N}^3 = {num_nodes:,} nodes\n")

print("1. Generating Training Snapshots...")

rng = np.random.RandomState(42)
N_train, N_val = 700, 140

train_freqs = rng.uniform(1.0, 3.0, size=(N_train, 3))
val_freqs   = rng.uniform(1.0, 3.0, size=(N_val,   3))

u_guess = jnp.zeros(num_nodes)

DATASET_PATH = SCRIPT_DIR / 'dataset.npz'

_need_generate = True
if DATASET_PATH.exists():
    _ds = np.load(DATASET_PATH)
    if _ds['U_train'].shape[0] == N_train and _ds['U_val'].shape[0] == N_val:
        print("   Loading cached dataset from dataset.npz...")
        U_train = jnp.array(_ds['U_train'])
        U_val   = jnp.array(_ds['U_val'])
        print(f"   Train: {U_train.shape}  |  Val: {U_val.shape}")
        _need_generate = False
    else:
        print(f"   Cache shape mismatch, regenerating...")

if _need_generate:
    U_train_list = []
    for i, (k1, k2, k3) in enumerate(train_freqs):
        U_train_list.append(full_order_fem_solver(get_F_3d(k1, k2, k3), u_guess))
        if (i + 1) % 20 == 0:
            print(f"   Solved {i+1}/{N_train} training snapshots...")

    U_val_list = []
    for k1, k2, k3 in val_freqs:
        U_val_list.append(full_order_fem_solver(get_F_3d(k1, k2, k3), u_guess))

    U_train = jnp.stack(U_train_list)
    U_val   = jnp.stack(U_val_list)
    np.savez(DATASET_PATH, U_train=np.array(U_train), U_val=np.array(U_val))
    print(f"   Train: {U_train.shape}  |  Val: {U_val.shape}  (saved)")

# ==========================================
# 4. Model Init
# ==========================================
RANK       = 512
HIDDEN_DIM = 256

model = ViTLinearCPAutoencoder(
    latent_dim     = k_dim,
    num_nodes      = num_nodes,
    patch_size     = 8,
    embed_dim      = 64,
    num_heads      = 4,
    num_enc_layers = 4,
    rank           = RANK,
    hidden_dim     = HIDDEN_DIM,
)

key    = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones(num_nodes))['params']

n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"   ViT-LinearCP parameters: {n_params:,}  "
      f"(latent_dim={k_dim}, rank={RANK}, hidden={HIDDEN_DIM})")

# ==========================================
# 5. Training
# ==========================================
BATCH_SIZE = 32
NUM_EPOCHS = 100_000
LOG_EVERY  = 2_000

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., peak_value=1e-3,
    warmup_steps=500, decay_steps=NUM_EPOCHS, end_value=1e-6
)
tx = optax.adamw(learning_rate=schedule, weight_decay=5e-4)

_ae_cfg = dict(
    arch='vit_linearcp_v3_skip_100k',
    k_dim=k_dim,
    rank=RANK,
    hidden_dim=HIDDEN_DIM,
    patch_size=8,
    embed_dim=64,
    num_heads=4,
    num_enc_layers=4,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    peak_lr=1e-3,
    end_lr=1e-6,
    warmup_steps=500,
    weight_decay=5e-4,
    N_train=N_train,
    N_val=N_val,
)
_fp = hashlib.md5(str(sorted(_ae_cfg.items())).encode()).hexdigest()[:12]
CKPT_PATH = SCRIPT_DIR / f'checkpoint_lincp_{_fp}.pkl'

def rel_l2_batch(u_true, u_pred):
    norms = jnp.linalg.norm(u_true - u_pred, axis=1)
    denom = jnp.linalg.norm(u_true, axis=1) + 1e-12
    return jnp.mean(norms / denom)

@jax.jit
def train_step(p, opt_st, batch):
    def loss_fn(weights):
        preds = jax.vmap(lambda u: model.apply({'params': weights}, u))(batch)
        return rel_l2_batch(batch, preds)
    loss, grads         = jax.value_and_grad(loss_fn)(p)
    updates, new_opt_st = tx.update(grads, opt_st, p)
    return optax.apply_updates(p, updates), new_opt_st, loss

@jax.jit
def eval_rec_err(p, batch):
    preds = jax.vmap(lambda u: model.apply({'params': p}, u))(batch)
    return rel_l2_batch(batch, preds)

if USE_CHECKPOINT and not SKIP_CHECKPOINT and CKPT_PATH.exists():
    print(f"\n2. --use-checkpoint: loading {CKPT_PATH.name}...")
    with open(CKPT_PATH, 'rb') as f:
        params = pickle.load(f)
    tr_err = float(eval_rec_err(params, U_train))
    vl_err = float(eval_rec_err(params, U_val))
    print(f"   Loaded. train rec {tr_err:.4e} | val rec {vl_err:.4e}")
else:
    if SKIP_CHECKPOINT:
        print(f"\n2. --skip-checkpoint: retraining from scratch...")
    elif USE_CHECKPOINT and not CKPT_PATH.exists():
        print(f"\n2. --use-checkpoint: no checkpoint found, training...")
    else:
        print(f"\n2. Training ViT-LinearCP ({NUM_EPOCHS} epochs, batch={BATCH_SIZE})...")
    opt_state = tx.init(params)
    n_train = len(U_train)
    t0      = time.perf_counter()
    key     = jax.random.PRNGKey(1)

    for epoch in range(NUM_EPOCHS + 1):
        key, sk = jax.random.split(key)
        idx     = jax.random.choice(sk, n_train, shape=(BATCH_SIZE,), replace=False)
        params, opt_state, loss = train_step(params, opt_state, U_train[idx])

        if epoch % LOG_EVERY == 0:
            tr_err = float(eval_rec_err(params, U_train))
            vl_err = float(eval_rec_err(params, U_val))
            print(f"   Epoch {epoch:5d} | loss {float(loss):.4e} | "
                  f"train rec {tr_err:.4e} | val rec {vl_err:.4e} | "
                  f"{time.perf_counter()-t0:.0f}s")

    print(f"   Training done in {time.perf_counter()-t0:.0f}s")
    print(f"   Saving checkpoint to {CKPT_PATH.name}...")
    with open(CKPT_PATH, 'wb') as f:
        pickle.dump(params, f)
    print(f"   Checkpoint saved.")

# ==========================================
# 6. NM-ROM Solver
# ==========================================
def make_rom_solver(model, params, K_operator, mask_vec, u_g_vec,
                    max_iters=10, tol=1e-6):
    def constrained_decode(z):
        u_hat = model.apply({'params': params}, z, method=model.decode)
        return mask_vec * u_hat + u_g_vec

    @jax.jit
    def rom_solve(z_init, F_vec):
        def body_fn(state):
            z, _, i = state
            u_pred, vjp_fn = jax.vjp(constrained_decode, z)
            R     = K_operator(u_pred) - F_vec
            r_red = vjp_fn(R)[0]

            def gn_op(dz):
                _, J_dz  = jax.jvp(constrained_decode, (z,), (dz,))
                K_J_dz   = K_operator(J_dz)
                return vjp_fn(K_J_dz)[0]

            delta_z, _ = jax_linalg.cg(gn_op, -r_red, tol=1e-3)
            z_new      = z + delta_z
            res_norm   = jnp.linalg.norm(r_red)
            return z_new, res_norm, i + 1

        def cond_fn(state):
            _, res_norm, i = state
            return (res_norm > tol) & (i < max_iters)

        u0       = constrained_decode(z_init)
        R0       = K_operator(u0) - F_vec
        _, vjp0  = jax.vjp(constrained_decode, z_init)
        r_red0   = vjp0(R0)[0]
        res0     = jnp.linalg.norm(r_red0)

        z_final, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (z_init, res0, 0)
        )
        return z_final, constrained_decode(z_final)

    return rom_solve

# ==========================================
# 7. Benchmark
# ==========================================
TEST_CASES = [
    (1.5, 2.3, 1.8),
    (2.1, 1.4, 2.7),
    (1.2, 2.8, 1.3),
    (2.5, 1.7, 2.2),
]
print(f"\n3. NM-ROM benchmark on {len(TEST_CASES)} unseen test cases...")

rom_solve = make_rom_solver(model, params, K_op_3d, mask, u_g, max_iters=8)

F_warm = get_F_3d(*TEST_CASES[0])
_ = jax_linalg.cg(K_op_3d, F_warm, x0=u_guess, tol=1e-6)[0].block_until_ready()
_, _ = rom_solve(jnp.zeros(k_dim), F_warm)
jax.effects_barrier()

fom_times, rom_times, rel_errors = [], [], []

for case_idx, (k1, k2, k3) in enumerate(TEST_CASES):
    F_test      = get_F_3d(k1, k2, k3)
    u_true_test = full_order_fem_solver(F_test, u_guess)

    z_init = jnp.zeros(k_dim)

    start_fom = time.time()
    u_fom, _  = jax_linalg.cg(K_op_3d, F_test, x0=u_guess, tol=1e-6)
    u_fom.block_until_ready()
    fom_time  = time.time() - start_fom

    start_rom            = time.time()
    z_final, u_pred_test = rom_solve(z_init, F_test)
    u_pred_test.block_until_ready()
    rom_time = time.time() - start_rom

    rel_l2 = float(jnp.linalg.norm(u_true_test - u_pred_test)
                   / jnp.linalg.norm(u_true_test))

    fom_times.append(fom_time)
    rom_times.append(rom_time)
    rel_errors.append(rel_l2)
    print(f"   Case {case_idx+1} k=({k1},{k2},{k3}): "
          f"FOM {fom_time:.4f}s  ROM {rom_time:.4f}s  "
          f"speedup {fom_time/rom_time:.2f}x  rel-L2 {rel_l2:.4e}")

avg_fom  = float(np.mean(fom_times))
avg_rom  = float(np.mean(rom_times))
avg_spup = avg_fom / avg_rom
avg_err  = float(np.mean(rel_errors))

print(f"\nAvg FOM time    : {avg_fom:.5f} s")
print(f"Avg ROM time    : {avg_rom:.5f} s   (speedup {avg_spup:.2f}x)")
print(f"Avg ROM rel L2  : {avg_err:.4e}")

# ==========================================
# 8. Midplane Slice Plot
# ==========================================
mid = N // 2
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

sl = np.s_[:, :, mid]

def plot_slice(ax, field, title):
    im = ax.imshow(np.array(field).reshape(N,N,N)[sl].T,
                   origin='lower', cmap='inferno',
                   extent=[0, L, 0, L], aspect='auto')
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)

k1, k2, k3 = TEST_CASES[-1]
F_last      = get_F_3d(k1, k2, k3)
u_true_last = full_order_fem_solver(F_last, u_guess)
_, u_pred_last = rom_solve(jnp.zeros(k_dim), F_last)

plot_slice(axs[0], u_true_last,  f"FOM  k=({k1:.1f},{k2:.1f},{k3:.1f})")
plot_slice(axs[1], u_pred_last,  f"NM-ROM LinearCP  k=({k1:.1f},{k2:.1f},{k3:.1f})")
err_field = jnp.abs(u_true_last - u_pred_last)
plot_slice(axs[2], err_field,    f"|Error|  Rel L2={rel_errors[-1]:.2e}")

plt.suptitle(f"3D Poisson ViT-LinearCP NM-ROM  —  z-midplane  (z={mid*dx:.2f})", fontsize=12)
plt.tight_layout()
out_path = SCRIPT_DIR / 'plots' / 'vitsep_linearcp_results.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")
print("\n=== ViT-LinearCP ROM COMPLETE ===")
