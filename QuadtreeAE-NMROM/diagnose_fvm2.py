"""Force a uniform 64x64 mesh by setting grad_thresh huge so depth-7 never
triggers. Then check FVM residual against the analytical u and FOM-CG residual.
This isolates whether the bug is in the basic stencil or only at T-junctions."""
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jspla
import f_to_u_nmrom as M

print(f"JAX devices: {jax.devices()}")
print(f"MIN_LEVEL={M.MIN_LEVEL} MAX_LEVEL={M.MAX_LEVEL}")

print("\n--- Uniform mesh test (grad_thresh=1e9 -> all leaves at depth=6) ---")
for k1, k2 in [(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.7, 1.9)]:
    qt = M.build_quadtree(k1, k2, grad_thresh=1e9)  # disable depth-7 refinement
    K_op, f_vec, u_ref, N, _, leaf_list = M.build_fvm_operator(qt)
    # Verify uniformity
    depths = [lf[3] for lf in leaf_list]
    sizes  = [lf[2] for lf in leaf_list]
    uniq_d = sorted(set(depths))
    print(f"  k=({k1},{k2}) N={N}  depths={uniq_d}  h_unique={sorted(set(sizes))}")

    # Direct residual
    K_u_ana = K_op(u_ref)
    res_d   = float(jnp.linalg.norm(K_u_ana - f_vec) / jnp.linalg.norm(f_vec))
    print(f"     ||K·u_ana - rhs||/||rhs|| = {res_d:.4e}")

    # FOM CG
    sol, _ = jspla.cg(K_op, f_vec, x0=jnp.zeros(N), tol=1e-12, maxiter=50000)
    sol = sol.block_until_ready()
    fom_res = float(jnp.linalg.norm(K_op(sol) - f_vec) / jnp.linalg.norm(f_vec))
    fom_err = float(jnp.linalg.norm(sol - u_ref) / jnp.linalg.norm(u_ref))
    print(f"     FOM CG: ||residual||/||rhs|| = {fom_res:.4e}  rel-L2(u_FOM, u_ana) = {fom_err:.4e}")
