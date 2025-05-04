def trace_execution(root):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(root)

    print("\n Trace of backpropagation (topological order):\n")

    for node in reversed(topo):
        op = node._op or "leaf"
        parents = [p.label or str(id(p)) for p in node._prev]
        print(f"→ Node '{node.label or str(id(node))}': op = {op}")
        print(f"   data = {node.data:.4f}, grad = {node.grad:.4f}")
        if parents:
           print(f"   came from: {', '.join(parents)}")
        print("") 
