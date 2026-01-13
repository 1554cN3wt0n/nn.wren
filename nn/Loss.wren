import "../data/Variable" for Variable

class Loss {
  calculate(pred, target) { Fiber.abort("Calculate not implemented.") }
}

class MSELoss is Loss {
  construct new() {}

  calculate(pred, target) {
    return Variable.mse(pred, target)
  }
}

