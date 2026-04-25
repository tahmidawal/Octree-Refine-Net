"""Diagnose FVM accuracy on adaptive vs uniform meshes by computing
the residual ||K · u_analytical - rhs|| directly. This tells us how good
the discretization is, separate from any solver issue."""
import numpy as np
import jax
import jax.numpy as jnp
import f_to_u_nmrom as M

print(f"JAX devices: {jax.devices()}")

def fvm_residual_test(k1, k2, label):
    qt = M.build_quadtree(k1, k2)
    K_op, f_vec, u_ref, N, _, leaf_list = M.build_fvm_operator(qt)

    # Apply K to the analytical solution at cell centers and compare to rhs.
    K_u_ana = K_op(u_ref)
    rhs     = f_vec
    res     = K_u_ana - rhs
    rel_res = float(np.linalg.norm(np.array(res)) / max(np.linalg.norm(np.array(rhs)), 1e-30))
    print(f"  {label}: N={N}  ||K·u_ana - rhs||/||rhs|| = {rel_res:.4e}")
    return rel_res

print("\nFVM truncation diagnostic — residual when applying K to the analytical u:")
print(f"  (Tree: |∇F|-adaptive, MIN_LEVEL={M.MIN_LEVEL}, MAX_LEVEL={M.MAX_LEVEL}, grad_thresh=0.5)")
for k1, k2 in [(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (1.0, 3.0), (2.7, 1.9)]:
    fvm_residual_test(k1, k2, f"k=({k1},{k2})")

# Also — solve K·u = rhs via direct CG and compare to u_ref
print("\nFOM CG residual on analytical-solution-discrepancy:")
import jax.scipy.sparse.linalg as jspla
for k1, k2 in [(1.5, 2.5)]:
    qt = M.build_quadtree(k1, k2)
    K_op, f_vec, u_ref, N, _, _ = M.build_fvm_operator(qt)
    sol, info = jspla.cg(K_op, f_vec, x0=jnp.zeros(N), tol=1e-10, maxiter=20000)
    sol = sol.block_until_ready()
    res = K_op(sol) - f_vec
    print(f"  k=({k1},{k2}) N={N}")
    print(f"    ||K·u_FOM - rhs||/||rhs|| = {float(jnp.linalg.norm(res)/jnp.linalg.norm(f_vec)):.4e}")
    print(f"    rel-L2(u_FOM, u_ana)      = {float(jnp.linalg.norm(sol - u_ref)/jnp.linalg.norm(u_ref)):.4e}")
