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

        next = self.data ** other
        next = Value(next)
        next._op = '**'
        next._prev = {self}

        def _backward_closure():
            self.grad += other * (self.data ** (other - 1)) * next.grad

        next._backward = _backward_closure

        return next
    
    def __neg__(self):
        next = -self.data 
        next = Value(next)
        next._op = '-'
        next._prev = {self}
        
        def _backward_closure():
            self.grad += -1 * next.grad
        
        next._backward = _backward_closure

        return next
    
    def __sub__(self, other):
        return self + -other
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __rsub__(self, other):
        return -self + other
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        sorted_list = []
        visited = set()

        def build_topo(node):
            if node in visited:          
                return                  
            visited.add(node)            
            for parent in node._prev:
                build_topo(parent)     
            sorted_list.append(node)    
                 
        build_topo(self)      
                                   
        self.grad = 1.0              

        for node in reversed(sorted_list): 
            node._backward()           