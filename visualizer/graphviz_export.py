from graphviz import Digraph

def trace(root):
    """Build every nodes and arches of the graph starting from root node"""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def draw_dot(root):
    """Visualize the computation graph using Graphviz"""
    dot = Digraph(format="png", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        label = f"{n.label} | data={n.data: .4f} | grad={n.grad: .4f}"
        dot.node(name=uid, label=label, shape="record")

        if n._op: # intermediate node (operation)
            op_id = uid + n._op
            dot.node(op_id, label=n._op)
            dot.edge(op_id, uid)
        
        for child in n._prev:
            dot.edge(str(id(child)), op_id)
    
    return dot