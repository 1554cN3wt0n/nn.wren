import "../nn/Model" for MLPModel
import "../data/Matrix" for Matrix
import "../data/Variable" for Variable
import "../nn/Layer" for LinearLayer, SequentialLayer, ReLuLayer, SigmoidLayer, SoftmaxLayer, LogSoftmaxLayer
import "../optim/SGD" for SGD
import "../nn/Loss" for CrossEntropyLoss, MSELoss, NLLLoss
import "../utils/Generator" for DataGenerator
import "../utils/Digits" for Digits
import "../utils/Dataloader" for DataLoader

// Load data
var loader = Digits.new()
loader.load("./resources/digits.csv")

// Setup DataLoader (Batch size 64, shuffle enabled)
var dataLoader = DataLoader.new(loader.features, loader.targets, 64, true)


// Define a 3-layer network: 64 inputs -> 32 hidden -> 10 outputs
var myModel = MLPModel.new([64, 32, 10])

// The optimizer just needs the model's parameters
var optimizer = SGD.new(myModel.parameters, 0.01)
var criterion = NLLLoss.new()

var batches = dataLoader.getBatches()

var epochLoss
for (epoch in 0...50){
  epochLoss = 0
  for (batch in batches) {
    // This is a Matrix of 64 pixels (e.g., 8 rows if batch size is 8)
    var inputData = batch["x"] 
  
    // This is a Matrix of 10 columns (one-hot encoded digits)
    var targetData = batch["y"]
  
    // --- The Training Step ---
    myModel.zeroGrad()
  
    var pred = myModel.forward(Variable.new(inputData))
    var loss = criterion.calculate(pred, Variable.new(targetData))
    epochLoss = epochLoss + loss.data[0, 0]
    loss.backward()
    optimizer.step()
  }

  if (epoch % 10 == 0) System.print("Epoch %(epoch) Average Loss: %(epochLoss / dataLoader.count)")
}
