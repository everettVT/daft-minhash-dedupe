
# Why original Connected Components was insufficient
(from gpt-5)
## Objective: Make our distributed star-contraction CC exactly match igraph’s connected components on the same edges.

### What broke parity
- Early stop: we stopped when a == b in a single iteration, which can terminate too soon.
- Partial labels: mapping only from u→v left out v-only nodes and left rep chains (u→rep→rep).
- Local minima: star steps can stabilize with multiple local representatives per true component.
- Validation pitfall: igraph built from non-contiguous np.uint64 labels can miscompare.

### Minimal additions that fixed it
1) Convergence on stabilized b  
   - Stop when consecutive small-star outputs stabilize: b_prev == b_next (as canonical sets).  
   - Prevents early termination.

2) Complete assignment over all nodes  
   - Build nodes = distinct(u ∪ v), join with per-u min(v), fill nulls with self.  
   - Ensures every node gets a label.

3) Path compression to roots  
   - Iteratively set rep ← rep_of_rep until fixed point.  
   - Removes label chains and splits.

4) Final global-min label propagation  
   - Over the undirected edge set, iteratively lower each node’s label to the min of its neighbors until stable.  
   - Collapses residual local minima to the component’s global minimum.

5) Robust igraph construction  
   - Map node IDs to contiguous Python int indices, build igraph on those, then map back.  
   - Avoids dtype/label ambiguity.

6) Validation by partition equality  
   - Group assignments by rep to sets of members and compare set-of-sets to igraph’s components.  
   - Label-invariant, exact.

7) Instrumentation (debuggable by design)  
   - Per-iteration metrics: edges(a/b), node count, distinct reps.  
   - Post checks: coverage gaps (u ∪ v vs assigned), multi-rep nodes, non-minimal groups.  
   - Mismatch previews with a few example components.

### Step-by-step narrative for the notebook
- We compute LSH bands, derive edges, and canonicalize to a directed high→low edge set b.  
- We alternate Large-Star and Small-Star until b stabilizes (consecutive b’s equal).  
- From the stabilized b, we construct an assignment for every node and compress labels to their roots.  
- We run a final label propagation over undirected edges to ensure every node’s label is the global minimum in its component.  
- For validation, we build an igraph using contiguous indices and compare the partition of nodes (set-of-sets).  
- We log indicators each iteration and post-convergence to catch regressions quickly.

### One-paragraph “why this works”
Star contraction reduces path lengths and merges local minima, but without careful stopping and label flattening it can stabilize with multiple local reps. Stabilizing on b (not a vs b), then compressing labels and propagating the global minimum over the undirected graph guarantees a single representative per true component. Using a robust igraph construction and comparing set partitions (not labels) gives an exact, label-invariant parity check.