class Vector: # Class
    def __init__(self, x, y): 
        self.x = x
        self.y = y

    def __add__(self, other):  # if you want plus two variables
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        # Scalar multiplication
        return Vector(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        # Scalar division
        if scalar != 0:  # Avoid division by zero
            return Vector(self.x / scalar, self.y / scalar)
        else:
            raise ValueError("Division by zero is undefined")

    def __len__(self):  # Define len() - length
        return 2

    def __repr__(self):  # Class printing
        return f"X: {self.x}, Y: {self.y}"

    def __call__(self):  # Calling object
        print(f"Someone searched me? {self.x} and {self.y} is here!")


v1 = Vector(10, 20)
v2 = Vector(10, 20)
v3 = v1 + v2
v3()
