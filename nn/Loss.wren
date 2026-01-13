import "../data/Variable" for Variable
import "../data/Matrix" for Matrix

class Loss {
  calculate(pred, target) { Fiber.abort("Calculate not implemented.") }
}

class MSELoss is Loss {
  construct new() {}

  calculate(pred, target) {
    return Variable.mse(pred, target)
  }
}

class CrossEntropyLoss is Loss {
  construct new() {}
  calculate(pred, target) { Variable.crossEntropy(pred, target) }
}

class NLLLoss is Loss {
  construct new() {}
  calculate(pred, target) {
    // NLL is simply -sum(target * input) assuming input is LogSoftmax
    var val = 0
    for (r in 0...pred.data.rows) {
      for (c in 0...pred.data.cols) {
        val = val - (target.data[r, c] * pred.data[r, c])
      }
    }
    var out = Variable.new(Matrix.new([[val / pred.data.cols]]))
    out.setCreator([pred], Fn.new { |gradOutput|
      var g = Matrix.empty(pred.data.rows, pred.data.cols)
      for (r in 0...pred.data.rows) {
        for (c in 0...pred.data.cols) g[r, c] = -target.data[r, c] * gradOutput[0, 0]
      }
      pred.accumulateGrad(g)
    })
    return out
  }
}
