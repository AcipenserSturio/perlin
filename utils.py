
import math

def pythagoras(x1, x2, y1, y2):
    return math.sqrt(abs(x1-x2)**2+abs(y1-y2)**2)

def greyscale(value):
    return [value, value, value]

def linear(value, x1, x2, y1, y2):
    return (y1-y2)*(value-x1)/(x1-x2) + y1

def sigmoid(value, bias=0):
    return math.tanh(math.tan(math.pi*value/2))

def semicircle(value):
    return math.sqrt(4-(value-1)**2)-1
