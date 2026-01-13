import "../data/Matrix" for Matrix
import "../data/Variable" for Variable
import "../nn/Layer" for LinearLayer, SequentialLayer, ReLuLayer, SigmoidLayer, SoftmaxLayer, LogSoftmaxLayer
import "../optim/SGD" for SGD
import "../nn/Loss" for CrossEntropyLoss, MSELoss, NLLLoss
import "../utils/Generator" for DataGenerator
import "../utils/Digits" for Digits
import "../utils/Dataloader" for DataLoader

// 1. Load data
var loader = Digits.new()
loader.load("./resources/digits.csv")

// 2. Setup DataLoader (Batch size 4, shuffle enabled)
var dataLoader = DataLoader.new(loader.features, loader.targets, 64, true)

// 3. Define the layers
var hiddenSize = 16

var net = SequentialLayer.new([
  // First Layer: 16 inputs -> 16 hidden neurons
  LinearLayer.new(64, hiddenSize), 
  ReLuLayer.new(),
  
  // Second Layer: 16 hidden -> 10 output classes
  LinearLayer.new(hiddenSize, 10),
  SigmoidLayer.new(), 
  // Final Activation for classification
  // LogSoftmaxLayer.new()
])

// 4. Setup the Training Components
// var criterion = NLLLoss.new()
var criterion = MSELoss.new()

var optimizer = SGD.new(net.parameters, 0.01) // Learning rate of 0.01

// 5. Training Loop with Mini-Batches
for (epoch in 1..100) {
  var epochLoss = 0
  var batches = dataLoader.getBatches() // Shuffles automatically

  for (batch in batches) {
    optimizer.zeroGrad()
    
    // Wrap batch matrices in Variables
    var x = Variable.new(batch["x"])
    var yTrue = Variable.new(batch["y"])
    
    // Forward -> Loss -> Backward -> Step
    var yPred = net.forward(x)
    var loss = criterion.calculate(yPred, yTrue)
    
    loss.backward()
    optimizer.step()
    
    epochLoss = epochLoss + loss.data[0, 0]
  }
  
  if (epoch % 10 == 0) System.print("Epoch %(epoch) Average Loss: %(epochLoss / dataLoader.count)")
}
