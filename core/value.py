import math

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data 
        self.grad = 0.0 
        self._backward = lambda : None
        self._prev = set(_children) 
        self._op = _op 
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    
    # OPERATION FOR Value
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward

        return out

    def __rtruediv__(self, other):  
        return Value(other) / self

    def __neg__(self):
        out = Value(-self.data, (self,), "neg")

        def _backward():
            self.grad += -1 * out.grad
        
        out._backward = _backward

        return out
    def __sub__(self, other): 
        return self + (-other)

    def __radd__(self, other): 
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other
    
    # FUNCTIONS
    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self, ), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def log(self):
        eps = 1e-12
        logval = math.log(max(self.data, eps))
        out = Value(logval, (self,), 'log')

        def _backward():
            self.grad += (1 / max(self.data, eps)) * out.grad

        out._backward = _backward
        return out

    
    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        x = self.data
        out = Value(x if x > 0 else 0, (self, ), "ReLu")

        def _backward():
            self.grad += (1.0 if x > 0 else 0.0) * out.grad
        out._backward = _backward

        return out
    
    def sin(self):
        x = self.data
        out = Value(math.sin(x), (self, ), "sin")

        def _backward():
            self.grad += math.cos(x) * out.grad
        out._backward = _backward

        return out
    
    def cos(self):
        x = self.data
        out = Value(math.cos(x), (self, ), "cos")

        def _backward():
            self.grad += -math.sin(x) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


