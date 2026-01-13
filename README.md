# About
This is yet another toy implementation of a simple deep learning library, but this time in [Wren](https://github.com/wren-lang/wren), a language that I have discovered just recently and I found it fascinating.

# Run me
You will need to first get clone the `Wren` repository and build it

```sh
git clone https://github.com/wren-lang/wren
cd wren
make -C projects/make
```

So now you will have the Wren runner under `wren/bin/wren_test`.

Now you can run any of the scripts under the `tests` directory.
```sh
wren_test tests/xor.wren
```

# Features
This implementation follows the same idea as PyTorch (formerly torch), it has a Matrix data type for matrix operations and a Variable type that wraps a Matrix to have a dual data called grad that have the gradient of the data and we can accumulate the gradient after we backpropagate some error.

Just to show how similar this is to PyTorch, here is the sample code for `tests/xor.wren`
```wren
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
```
