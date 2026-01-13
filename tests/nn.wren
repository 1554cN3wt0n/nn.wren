import "../data/Matrix" for Matrix
import "../data/Variable" for Variable
import "../optim/SGD" for SGD
import "../nn/Layer" for SequentialLayer, LinearLayer, ReLuLayer, SigmoidLayer

// Define a network: 2 inputs -> 4 hidden units (ReLU) -> 1 output (Sigmoid)
var net = SequentialLayer.new([
  LinearLayer.new(2, 4),
  ReLuLayer.new(),
  LinearLayer.new(4, 1),
  SigmoidLayer.new()
])

// Dummy Input: [0.5, -0.2]
var input = Variable.new(Matrix.new([[0.5], [-0.2]]))

// Forward Pass
var output = net.forward(input)
System.print("Network Prediction:")
System.print(output.data)

// Backward Pass (Calculates gradients for all W and B in the chain)
output.backward()

System.print("Gradient for first layer weights (first 2 elements):")
var firstLayerW = net.parameters[0]
System.print(firstLayerW.grad)
