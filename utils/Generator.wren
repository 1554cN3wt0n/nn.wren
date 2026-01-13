import "random" for Random
import "../data/Matrix" for Matrix

class DataGenerator {
  static generateDonut(count) {
    var rng = Random.new()
    var inputs = Matrix.empty(count, 2)
    var targets = Matrix.empty(count, 2)

    for (i in 0...count) {
      var r = (i < count / 2) ? rng.float() * 0.4 : 0.7 + rng.float() * 0.3
      var theta = rng.float() * 2 * 3.14159
      var x = r * theta.cos
      var y = r * theta.sin
      
      inputs[i, 0] = x
      inputs[i, 1] = y
      // One-hot encode: [1, 0] for inner, [0, 1] for outer
      if(i < count / 2) {
          targets[i, 0] = 1
          targets[i, 1] = 0
      }else {
           targets[i, 0] = 0
          targets[i, 1] = 1
      }
    }
    return {"x": inputs, "y": targets}
  }
}
