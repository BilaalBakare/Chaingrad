import math

class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._op = ''
        self._prev = set()
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(Data = {self.data}, grad = {self.grad})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        next = self.data + other.data
        next = Value(next)
        next._op = '+'
        next._prev = {self, other}
        
        def _backward_closure():
            self.grad += 1 * next.grad
            other.grad += 1 * next.grad
        
        next._backward = _backward_closure

        return next
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        next = self.data * other.data
        next = Value(next)
        next._op = '*'
        next._prev = {self, other}
        
        def _backward_closure():
            self.grad += other.data * next.grad
            other.grad += self.data * next.grad
        
        next._backward = _backward_closure

        return next
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        next = self.data ** other.data
        next = Value(next)
        next._op = '**'
        next._prev = {self, other}
        
        def _backward_closure():
            self.grad += other.data * (self.data ** (other.data - 1)) * next.grad
            other.grad += next.data * math.log(self.data) * next.grad
        
        next._backward = _backward_closure

        return next