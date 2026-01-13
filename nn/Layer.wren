import "random" for Random
import "../data/Matrix" for Matrix
import "../data/Variable" for Variable 

class Layer {
  forward(input) { Fiber.abort("Forward not implemented.") }
  parameters { [] } // Returns a list of Variables to be optimized
}

class LinearLayer is Layer {
  construct new(inFeatures, outFeatures) {
    var rng = Random.new()
    // Initialize weights with small random values (Xavier-lite)
    var wData = Matrix.empty(inFeatures, outFeatures)
    for (r in 0...inFeatures) {
      for (c in 0...outFeatures) wData[r, c] = rng.float() * 0.2 - 0.1
    }
    _w = Variable.new(wData)
    
    // Initialize bias with zeros
    _b = Variable.new(Matrix.empty(1, outFeatures))

  }

  parameters { [_w, _b] }

  forward(input) {
    // Y = W * X + B
    var wx = Variable.matmul(input, _w)
    return Variable.add(wx, Variable.matmul(Variable.ones(wx.data.rows, 1), _b))
  }
}

class SigmoidLayer is Layer {
  construct new() {}
  forward(input) { Variable.sigmoid(input) }
  parameters { [] }
}

class ReLuLayer is Layer {
  construct new() {}
  forward(input) { Variable.relu(input) }
  parameters { [] }
}

class SequentialLayer is Layer {
  construct new(layers) {
    _layers = layers
  }

  parameters {
    var params = []
    for (layer in _layers) params.addAll(layer.parameters)
    return params
  }

  forward(input) {
    var current = input
    for (layer in _layers) {
      current = layer.forward(current)
    }
    return current
  }
}

class SoftmaxLayer is Layer {
  construct new() {}
  forward(input) { Variable.softmax(input) }
  parameters { [] }
}

class LogSoftmaxLayer is Layer {
  construct new() {}
  forward(input) { Variable.logSoftmax(input) }
  parameters { [] }
}
