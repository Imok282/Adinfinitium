# ADR-001: JAX for the 10,000-mission swarm simulator

**Status:** Accepted
**Date:** 2026-11-14
**Deciders:** Imok (owner sim + swarm), team

## Context
Each generation the swarm flies 10,000 missions of about 2,400 time steps each, repeated for
hundreds of generations overnight on one RTX 4090. The evolution library we want, TensorNEAT,
is built on JAX. The world-model training (U-Net, diffusion, neural field) is easier in PyTorch.

## Decision
Write the swarm simulator in pure JAX: a gymnax-style pure `reset`/`step`, `vmap` over
missions, and `lax.scan` over time. World-model training stays in PyTorch, in a separate venv.
The two sides exchange `.npz` files only.

## Options Considered

### Option A: JAX (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Medium — functional style, no Python control flow inside the step |
| Cost | Free; one GPU |
| Scalability | Excellent — the whole population stays on the GPU; `pmap` later |
| Team familiarity | Medium (Imok); vibe-coders don't touch it |

**Pros:** native for TensorNEAT; one compiled kernel for the whole generation; deterministic PRNG keys, so AI and classic guidance can fly *identical* missions.
**Cons:** two CUDA stacks on the 4090; long first compile (~5 s); harder to debug.

### Option B: PyTorch batched sim
| Dimension | Assessment |
|---|---|
| Complexity | Low–Medium |
| Cost | Free |
| Scalability | Good |
| Team familiarity | Higher |

**Pros:** one framework for everything.
**Cons:** no TensorNEAT; we would need our own GPU NEAT or would lose the "brain grows" story.

### Option C: NumPy / Basilisk-only
**Cons:** 10k missions × 300 generations is out of reach on a CPU within the hackathon.

### Option D: Isaac Sim / Brax
**Cons:** heavy installs on a no-Claude PC; overkill for 3-DOF terminal descent.

## Trade-off Analysis
TensorNEAT forces JAX for the evolution loop. Keeping the sim in the same framework avoids a
GPU↔host copy every step, and that copy would dominate the run time. Splitting the venvs is
the price of keeping PyTorch for the world models.

## Consequences
- Easier: identical-key comparisons, massive parallelism, `vmap` over terrain versions.
- Harder: anything stateful or branching must be written with `jnp.where`/`select`.
- Revisit: 6-DOF dynamics; orbit phases inside JAX.

## Action Items
1. [x] 3-DOF env + classic guidance + vmap rollout (`sim_jax/`), tests on CPU
2. [ ] `scripts.bench_swarm --n 10000` on the 4090 (KAN-101)
3. [ ] TensorNEAT problem wrapper (KAN-105)
