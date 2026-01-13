import "../data/Matrix" for Matrix
import "../data/Variable" for Variable
import "../nn/Layer" for LinearLayer, SequentialLayer, ReLuLayer, SigmoidLayer
import "../optim/SGD" for SGD
import "../nn/Loss" for MSELoss

// 1. Setup Data (XOR inputs and labels)
var inputs = [
  Matrix.new([[0], [0]]),
  Matrix.new([[0], [1]]),
  Matrix.new([[1], [0]]),
  Matrix.new([[1], [1]])
]
var targets = [
  Matrix.new([[0]]),
  Matrix.new([[1]]),
  Matrix.new([[1]]),
  Matrix.new([[0]])
]

// 2. Define Network (2 -> 4 -> 1)
var net = SequentialLayer.new([
  LinearLayer.new(2, 4),
  ReLuLayer.new(),
  LinearLayer.new(4, 1),
  SigmoidLayer.new()
])

var optimizer = SGD.new(net.parameters, 0.1)
var lossfn = MSELoss.new()

// 3. Training Loop
System.print("Starting Training...")
for (epoch in 0..2000) {
  var totalLoss = 0
  
  for (i in 0...inputs.count) {
    optimizer.zeroGrad()
    
    // Forward
    var x = Variable.new(inputs[i])
    var y_true = Variable.new(targets[i])
    var y_pred = net.forward(x)
    
    // Loss
    var loss = lossfn.calculate(y_pred, y_true)
    totalLoss = totalLoss + loss.data[0, 0]
    
    // Backward & Optimize
    loss.backward()
    optimizer.step()
  }
  
  if (epoch % 500 == 0) System.print("Epoch %(epoch), Loss: %(totalLoss / 4)")
}

// 4. Test the results
System.print("\nFinal Predictions:")
for (i in 0...inputs.count) {
  var out = net.forward(Variable.new(inputs[i]))
  System.print("Input: %(inputs[i].data[0])%(inputs[i].data[1]) -> Predict: %(out.data[0, 0])")
}
