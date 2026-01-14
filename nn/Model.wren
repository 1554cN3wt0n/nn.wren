import "./Layer" for LinearLayer, ReLuLayer, LogSoftmaxLayer

class Model {
  // Abstract method: must be implemented by child classes
  forward(input) { Fiber.abort("Model must implement forward(input)") }

  // Returns all parameters (Weights/Biases) from all layers
  parameters {
    var params = []
    for (layer in layers) {
      params.addAll(layer.parameters)
    }
    return params
  }

  // Resets all gradients in the model to zero
  zeroGrad() {
    for (p in parameters) p.zeroGrad()
  }

  // Abstract getter: child class provides the list of Layer objects
  layers { [] }
}

class MLPModel is Model {
  // Construct the network by defining dimensions (e.g., [16, 32, 10])
  construct new(dims) {
    _layers = []
    for (i in 0...dims.count - 1) {
      var inDim = dims[i]
      var outDim = dims[i+1]

      // Add a Linear Layer
      _layers.add(LinearLayer.new(inDim, outDim))

      // Add ReLU for all hidden layers, but not the last one
      if (i < dims.count - 2) {
        _layers.add(ReLuLayer.new())
      }
    }
    // Final layer usually gets Softmax/LogSoftmax
    _layers.add(LogSoftmaxLayer.new())
  }

  // Implementation of parameters from the base class
  layers { _layers }

  // Logic to pass data through all layers
  forward(input) {
    var current = input
    for (layer in _layers) {
      current = layer.forward(current)
    }
    return current
  }
}
