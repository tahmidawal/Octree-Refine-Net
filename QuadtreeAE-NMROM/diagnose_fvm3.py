"""Test FVM on uniform mesh with INTEGER k. The 'analytical' formula
sin(k·π·x)·sin(k·π·y)/((k²+k²)π²) only satisfies u=0 on [0,1]² when k is integer.
For non-integer k it violates the boundary condition, so FOM (which enforces u=0)
diverges from it."""
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jspla
import f_to_u_nmrom as M

print(f"JAX devices: {jax.devices()}")
print("\n--- Uniform 64x64 mesh, INTEGER k1, k2 ---")
for k1, k2 in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
    qt = M.build_quadtree(float(k1), float(k2), grad_thresh=1e9)
    K_op, f_vec, u_ref, N, _, _ = M.build_fvm_operator(qt)

    K_u_ana = K_op(u_ref)
    res_d   = float(jnp.linalg.norm(K_u_ana - f_vec) / jnp.linalg.norm(f_vec))
    print(f"  k=({k1},{k2}) N={N}  ||K·u_ana - rhs||/||rhs|| = {res_d:.4e}")

    sol, _ = jspla.cg(K_op, f_vec, x0=jnp.zeros(N), tol=1e-12, maxiter=50000)
    sol = sol.block_until_ready()
    fom_res = float(jnp.linalg.norm(K_op(sol) - f_vec) / jnp.linalg.norm(f_vec))
    fom_err = float(jnp.linalg.norm(sol - u_ref) / jnp.linalg.norm(u_ref))
    print(f"     FOM CG: ||residual||/||rhs|| = {fom_res:.4e}  rel-L2(u_FOM, u_ana) = {fom_err:.4e}")
