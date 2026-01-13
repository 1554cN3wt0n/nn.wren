import "../data/Matrix" for Matrix
import "../data/Variable" for Variable
import "../optim/SGD" for SGD

// 1. Initialize Weight and Input
var W = Variable.new(Matrix.new([[0.5]])) // Random start weight
var x = Variable.new(Matrix.new([[1.0]])) // Input
var target = 0.0                          // We want the result to be 0

var optimizer = SGD.new([W], 0.1)

System.print("Starting Training...")

for (epoch in 1..20) {
  optimizer.zeroGrad()

  // Forward Pass: Prediction = Sigmoid(W * x)
  var prediction = Variable.sigmoid(Variable.matmul(W, x))

  // Calculate Loss: (prediction - target)^2
  // (We'll do a simple manual loss gradient for the root)
  var diff = prediction.data[0, 0] - target
  var loss = diff * diff

  // Backward Pass:
  // Set the gradient of the prediction node to start the chain
  // dLoss/dPrediction = 2 * (prediction - target)
  prediction.grad[0, 0] = 2 * diff
  prediction.backward()

  // Update Weights
  optimizer.step()

  if (epoch % 5 == 0) {
    System.print("Epoch %(epoch): Loss = %(loss), W = %(W.data[0, 0])")
  }
}

System.print("\nFinal Prediction: %(Variable.sigmoid(Variable.matmul(W, x)).data[0, 0])")


