import "./Matrix" for Matrix

class Variable {
  construct new(data) {
    _data = data
    // Initialize gradient with zeros (same shape as data)
    _grad = Matrix.empty(data.rows, data.cols)
    _parents = []
    _backwardFunc = null
  }

  // Getters
  data { _data }
  grad { _grad }
  
  // Setters
  grad=(m) { _grad = m }

  // Reset gradients to zero before a new backward pass
  zeroGrad() {
    _grad = Matrix.empty(_data.rows, _data.cols)
  }

  // Define how this Variable was created
  setCreator(parents, func) {
    _parents = parents
    _backwardFunc = func
  }

  // The Backpropagation trigger
  backward() {
    // If this is the starting node (like Loss), set grad to 1s
    if (_grad.rows == _data.rows && _grad.cols == _data.cols) {
      // For a scalar loss, grad is usually 1.0
      // For simplicity, we initialize the root grad to 1.0 if empty
      var allZeros = true
      for (r in 0..._grad.rows) {
        for (c in 0..._grad.cols) {
          if (_grad[r, c] != 0) allZeros = false
        }
      }
      if (allZeros) {
        for (r in 0..._grad.rows) {
          for (c in 0..._grad.cols) _grad[r, c] = 1.0
        }
      }
    }

    // Call the specific backward logic for the op that created this
    if (_backwardFunc != null) {
      _backwardFunc.call(_grad)
    }

    // Recursively call backward for parents
    for (p in _parents) {
      p.backward()
    }
  }

  // --- Operations with Autograd ---
 static add(a, b) {
    var out = Variable.new(a.data.add(b.data))
    out.setCreator([a, b], Fn.new { |gradOutput|
      // dL/dA = dL/dOut * 1
      // dL/dB = dL/dOut * 1
      a.accumulateGrad(gradOutput)
      b.accumulateGrad(gradOutput)
    })
    return out
  }

  static sub(a, b) {
    var out = Variable.new(a.data.sub(b.data))
    out.setCreator([a, b], Fn.new { |gradOutput|
      // dL/dA = dL/dOut * 1
      // dL/dB = dL/dOut * -1
      a.accumulateGrad(gradOutput)
      
      var negGrad = Matrix.empty(gradOutput.rows, gradOutput.cols)
      for (r in 0...gradOutput.rows) {
        for (c in 0...gradOutput.cols) negGrad[r, c] = -gradOutput[r, c]
      }
      b.accumulateGrad(negGrad)
    })
    return out
  }

  static times(a, b) {
    var out = Variable.new(a.data.times(b.data))
    out.setCreator([a, b], Fn.new { |gradOutput|
      // dL/dA = dL/dOut * B
      // dL/dB = dL/dOut * A
      a.accumulateGrad(gradOutput.times(b.data))
      b.accumulateGrad(gradOutput.times(a.data))
    })
    return out
  }

  static div(a, b) {
    var out = Variable.new(a.data.div(b.data))
    out.setCreator([a, b], Fn.new { |gradOutput|
      // dL/dA = gradOutput / B
      a.accumulateGrad(gradOutput.div(b.data))
      
      // dL/dB = gradOutput * (-A / B^2)
      var gradB = Matrix.empty(a.data.rows, a.data.cols)
      for (r in 0...a.data.rows) {
        for (c in 0...a.data.cols) {
          var aVal = a.data[r, c]
          var bVal = b.data[r, c]
          gradB[r, c] = gradOutput[r, c] * (-aVal / (bVal * bVal))
        }
      }
      b.accumulateGrad(gradB)
    })
    return out
  }
  // Matrix Multiplication: C = A * B
  static matmul(a, b) {
    var out = Variable.new(a.data.matmul(b.data))
    
    // Define the gradient logic:
    // dL/dA = dL/dC * B^T
    // dL/dB = A^T * dL/dC
    out.setCreator([a, b], Fn.new { |gradOutput|
      var gradA = gradOutput.matmul(b.data.transpose())
      var gradB = a.data.transpose().matmul(gradOutput)
      
      // Accumulate gradients
      a.accumulateGrad(gradA)
      b.accumulateGrad(gradB)
    })
    
    return out
  }
  static relu(a) {
    var outData = a.data.map(Fn.new { |x| x > 0 ? x : 0 })
    var out = Variable.new(outData)
    
    out.setCreator([a], Fn.new { |gradOutput|
      var gradA = Matrix.empty(a.data.rows, a.data.cols)
      for (r in 0...a.data.rows) {
        for (c in 0...a.data.cols) {
          // Gradient is 1 if input was > 0, else 0
          gradA[r, c] = (a.data[r, c] > 0) ? gradOutput[r, c] : 0
        }
      }
      a.accumulateGrad(gradA)
    })
    return out
  }

  static sigmoid(a) {
    var outData = a.data.map(Fn.new { |x| 1 / (1 + (-x).exp) })
    var out = Variable.new(outData)
    
    out.setCreator([a], Fn.new { |gradOutput|
      var gradA = Matrix.empty(a.data.rows, a.data.cols)
      for (r in 0...a.data.rows) {
        for (c in 0...a.data.cols) {
          // Sigmoid derivative: f(x) * (1 - f(x))
          var s = out.data[r, c]
          gradA[r, c] = gradOutput[r, c] * (s * (1 - s))
        }
      }
      a.accumulateGrad(gradA)
    })
    return out
  }
  
  static mse(pred, target) {
    var diff = Variable.sub(pred, target)
    var square = Variable.times(diff, diff)

    // Calculate mean of squared differences
    var sum = 0
    var count = square.data.rows * square.data.cols
    for (r in 0...square.data.rows) {
      for (c in 0...square.data.cols) sum = sum + square.data[r, c]
    }

    var out = Variable.new(Matrix.new([[sum / count]]))
    out.setCreator([pred, target], Fn.new { |gradOutput|
      var factor = 2 / count
      var gradInput = Matrix.empty(pred.data.rows, pred.data.cols)

      for (r in 0...pred.data.rows) {
        for (c in 0...pred.data.cols) {
          // dMSE/dPred = 2/n * (pred - target)
          gradInput[r, c] = gradOutput[r, c] * factor * (pred.data[r, c] - target.data[r, c])
        }
      }
      pred.accumulateGrad(gradInput)
      // We usually don't need the gradient of the target (labels),
      // but we could add target.accumulateGrad here if doing GANs.
    })
    return out
  }

  // Helper to accumulate gradients (crucial for nodes used multiple times)
  accumulateGrad(newGrad) {
    for (r in 0..._grad.rows) {
      for (c in 0..._grad.cols) {
        _grad[r, c] = _grad[r, c] + newGrad[r, c]
      }
    }
  }

}
