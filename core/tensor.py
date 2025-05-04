from .value import Value

class Tensor:
    def __init__(self, data, label="T"):
        """
        data: lista di liste (matrice), o lista (vettore), o scalare
        Ogni elemento viene convertito in un Value se non lo è già
        """
        self.data = self._wrap(data, label)
        self.shape = self._infer_shape(self.data)

    def _wrap(self, x, label, path=()):
        if isinstance(x, (int, float)):
            full_label = f"{label}{''.join(f'[{i}]' for i in path)}"
            return Value(x, label=full_label)
        elif isinstance(x, Value):
            return x
        elif isinstance(x, list):
            return [self._wrap(el, label, path + (i,)) for i, el in enumerate(x)]
        else:
            raise TypeError(f"Tipo non supportato: {type(x)}")
        
    def _infer_shape(self, x):
        if isinstance(x, Value):
            return ()
        elif isinstance(x, list):
            if len(x) == 0:
                return (0,)
            return (len(x),) + self._infer_shape(x[0])

    def __repr__(self):
        def format_tensor(data, depth=0):
            if isinstance(data, Value):
                return f"{data.data:.4f}"
            inner = ", ".join(format_tensor(x, depth + 1) for x in data)
            return "[" + inner + "]"
        
        return format_tensor(self.data)
    
    def parameters(self):
        def collect(v):
            if isinstance(v, Value):
                return [v]
            res = []
            for elem in v:
                res.extend(collect(elem))
            return res
        return collect(self.data)
    
    def _apply_recursive(self, a, b, op):
        if isinstance(a, Value):
            return op(a, b)
        return [self._apply_recursive(x, y, op) for x, y in zip(a, b)]

    def _apply_scalar(self, a, op):
        if isinstance(a, Value):
            return op(a)
        return [self._apply_scalar(x, op) for x in a]

    def __add__(self, other):
        return Tensor(self._apply_recursive(self.data, other.data, lambda a, b: a + b))

    def __sub__(self, other):
        return Tensor(self._apply_recursive(self.data, other.data, lambda a, b: a - b))

    def __mul__(self, other):
        return Tensor(self._apply_recursive(self.data, other.data, lambda a, b: a * b))

    def __truediv__(self, other):
        return Tensor(self._apply_recursive(self.data, other.data, lambda a, b: a / b))

    def __pow__(self, exp):
        assert isinstance(exp, (int, float))
        return Tensor(self._apply_scalar(self.data, lambda a: a ** exp))

    def __neg__(self):
        return Tensor(self._apply_scalar(self.data, lambda a: -a))
    
    def tanh(self):
        return Tensor(self._apply_scalar(self.data, lambda v: v.tanh()))

    def exp(self):
        return Tensor(self._apply_scalar(self.data, lambda v: v.exp()))

    def relu(self):
        return Tensor(self._apply_scalar(self.data, lambda v: v.relu()))

    def log(self):
        return Tensor(self._apply_scalar(self.data, lambda v: v.log()))
    
    def __matmul__(self, other):
        assert len(self.shape) == 2 and len(other.shape) == 2, "Solo Tensor 2D supportati per @"
        assert self.shape[1] == other.shape[0], "Dimensioni incompatibili"
        result = []
        for row in self.data:
            new_row = []
            for col in zip(*other.data):  
                dot = sum([a * b for a, b in zip(row, col)])
                new_row.append(dot)
            result.append(new_row)
        return Tensor(result)
    
    def det(self):
        assert len(self.shape) == 2, "Determinante definito solo per Tensor 2D"
        n, m = self.shape
        assert n == m, "Det definito solo per matrici quadrate"

        def _minor(mat, row, col):
            return [
                [v for j, v in enumerate(r) if j != col]
                for i, r in enumerate(mat) if i != row
            ]

        def _det(matrix):
            if len(matrix) == 1:
                return matrix[0][0]
            if len(matrix) == 2:
                a, b = matrix[0]
                c, d = matrix[1]
                return a * d - b * c

            total = Value(0.0)
            for j, elem in enumerate(matrix[0]):
                sign = (-1) ** j
                minor = _minor(matrix, 0, j)
                total += elem * sign * _det(minor)
            return total

        return _det(self.data)
    
    def inv_gauss_jordan(self):
        assert len(self.shape) == 2, "Inversa solo per matrici 2D"
        n, m = self.shape
        assert n == m, "Matrice non quadrata"

        A = [[v for v in row] for row in self.data]

        I = [[Value(1.0 if i == j else 0.0, label=f"I[{i},{j}]") for j in range(n)] for i in range(n)]

        for i in range(n):
            pivot = A[i][i]
            assert pivot.data != 0, "Pivot nullo, la matrice non è invertibile"

            # Normalizza la riga i
            for j in range(n):
                A[i][j] = A[i][j] / pivot
                I[i][j] = I[i][j] / pivot

            # Elimina le altre righe
            for k in range(n):
                if k == i:
                    continue
                factor = A[k][i]
                for j in range(n):
                    A[k][j] = A[k][j] - factor * A[i][j]
                    I[k][j] = I[k][j] - factor * I[i][j]

        return Tensor(I)
    
    def flatten(self):
        def flatten_data(x):
            if isinstance(x, Value):
                return [x]
            res = []
            for el in x:
                res.extend(flatten_data(el))
            return res
        flat = flatten_data(self.data)
        return Tensor([flat])
    
    def reshape(self, shape):
        flat = self.flatten().data[0]
        total = 1
        for dim in shape:
            total *= dim
        assert total == len(flat), "Mismatch tra elementi e nuova shape"

        def build(shape, values):
            if len(shape) == 1:
                return [values.pop(0) for _ in range(shape[0])]
            return [build(shape[1:], values) for _ in range(shape[0])]
        
        reshaped = build(list(shape), flat.copy())
        return Tensor(reshaped)
    
    def permute(self, *dims):
        assert len(self.shape) == len(dims), "permute: dimensioni non corrispondenti"

        import itertools

        def recursive_index(data, indices):
            for i in indices:
                data = data[i]
            return data
        
        def build(shape, dims, original):
            if not shape:
                return original
            result = []
            for i in range(shape[0]):
                sub = build(shape[1:], dims, original)
                result.append(sub)
            return result
        
        all_indices = list(itertools.product(*[range(d) for d in self.shape]))
        permuted = {}

        for i in all_indices:
            src = recursive_index(self.data, i)
            tgt_i = tuple(i[d] for d in dims)
            permuted[tgt_i] = src
        
        def reconstruct(shape, level=0, prefix=()):
            if level == len(shape):
                return permuted[prefix]
            return [reconstruct(shape, level + 1, prefix + (i, )) for i in range(shape[level])]
        
        new_shape = tuple(self.shape[d] for d in dims)
        data = reconstruct(new_shape)
        return Tensor(data)
    
    def __getitem__(self, idx):
        # Converti indice singolo in tupla
        if not isinstance(idx, tuple):
            idx = (idx,)

        def recursive_slice(data, idx):
            if len(idx) == 0:
                return data
            i = idx[0]
            if isinstance(i, slice):
                return [recursive_slice(d, idx[1:]) for d in data[i]]
            else:
                return recursive_slice(data[i], idx[1:])
        
        result = recursive_slice(self.data, idx)

        # Se è ancora una struttura annidata, ritorna Tensor
        if isinstance(result, list):
            return Tensor(result)
        return result 
    
    @property
    def T(self):
        assert len(self.shape) == 2, "La trasposta è definita solo per Tensor 2D"
        transposed = list(zip(*self.data)) 
        transposed = [list(row) for row in transposed]
        return Tensor(transposed)
    

