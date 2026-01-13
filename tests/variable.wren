import "../data/Matrix" for Matrix
import "../data/Variable" for Variable

var W_data = Matrix.new([
  [0.5, -0.1],
  [0.2,  0.8]
])
var X_data = Matrix.new([
  [1.0],
  [2.0]
])

// Wrap in Variables
var W = Variable.new(W_data)
var X = Variable.new(X_data)

// Forward Pass: Y = W * X
var Y = Variable.matmul(W, X)

System.print("Output Y data:")
System.print(Y.data)

// Backward Pass
Y.backward()

System.print("Gradient of W (dL/dW):")
System.print(W.grad)

System.print("Gradient of X (dL/dX):")
System.print(X.grad)

var a = Variable.new(Matrix.new([[2, 4]]))
var b = Variable.new(Matrix.new([[1, 3]]))

// Equation: (a + b) * a
var sum = Variable.add(a, b)
var result = Variable.times(sum, a)

result.backward()

System.print("Result data: %(result.data)") // Should be [[(2+1)*2, (4+3)*4]] -> [[6, 28]]
System.print("Grad of a: %(a.grad)")

