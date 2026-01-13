class SGD {
  construct new(parameters, learningRate) {
    _params = parameters
    _lr = learningRate
  }

  step() {
    for (p in _params) {
      for (r in 0...p.data.rows) {
        for (c in 0...p.data.cols) {
          // Move data: data = data - (lr * gradient)
          p.data[r, c] = p.data[r, c] - (_lr * p.grad[r, c])
        }
      }
    }
  }

  zeroGrad() {
    for (p in _params) p.zeroGrad()
  }
}
