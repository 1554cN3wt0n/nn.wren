import "../data/Matrix" for Matrix
import "../data/Variable" for Variable
import "../nn/Layer" for LinearLayer, SequentialLayer, ReLuLayer, SigmoidLayer, SoftmaxLayer, LogSoftmaxLayer
import "../optim/SGD" for SGD
import "../nn/Loss" for CrossEntropyLoss, MSELoss, NLLLoss
import "../utils/Generator" for DataGenerator

// 1. Prepare Data
var dataset = DataGenerator.generateDonut(100)
var trainX = dataset["x"]
var trainY = dataset["y"]

// 2. Define a Deep Neural Network
// Input(2) -> Hidden(8) -> Hidden(8) -> Output(2)
var net = SequentialLayer.new([
  LinearLayer.new(2, 8),
  ReLuLayer.new(),
  LinearLayer.new(8, 8),
  ReLuLayer.new(),
  LinearLayer.new(8, 2),
  // SigmoidLayer.new(),
  // SoftmaxLayer.new(),
  LogSoftmaxLayer.new(),
])

// var criterion = MSELoss.new()
// var criterion = CrossEntropyLoss.new()
var criterion = NLLLoss.new()
var optimizer = SGD.new(net.parameters, 0.01)

// 3. Training Loop
System.print("Training on Donut Clustering...")
System.print("Epoch | Average Loss")
System.print("--------------------")

for (epoch in 1..500) {
  var epochLoss = 0
  
  optimizer.zeroGrad()
  
  // Forward Pass
  var x = Variable.new(trainX)
  var yTrue = Variable.new(trainY)
  var yPred = net.forward(x)
  
  // Calculate Loss
  var loss = criterion.calculate(yPred, yTrue)
  epochLoss = loss.data[0, 0]
  
  // Backward Pass
  loss.backward()
  
  // Update Weights
  optimizer.step()
 
  
  if (epoch % 250 == 0 || epoch == 1) {
    var avgLoss = (epochLoss / trainX.rows).toString
    // Clean up the string for the terminal
    if (avgLoss.count > 6) avgLoss = avgLoss[0...6]
    System.print("%(epoch) | %(avgLoss)")
  }
}

// 4. Evaluation / Visualization
System.print("\nTesting Classifications:")
var correct = 0
var sample = Matrix.empty(4, 2)
var idx = 0
for (i in [0, 10, 60, 90]) { // Sample a few points
    sample[idx, 0] = trainX[i, 0]
    sample[idx, 1] = trainX[i, 1]
    idx = idx + 1
}

var x = Variable.new(sample)
var pred = net.forward(x)

idx = 0
for (i in [0, 10, 60, 90]){
  var actual = (trainY[i, 0] == 1) ? "Inner" : "Outer"
  var predicted = (pred.data[idx, 0] > pred.data[idx, 1]) ? "Inner" : "Outer"
  idx = idx + 1 
  System.print("Point at [%(trainX[i, 0]), %(trainX[i, 1])] -> Actual: %(actual), Pred: %(predicted)")
}

