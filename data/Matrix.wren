class Matrix {
  // Create a matrix from a 2D list
  construct new(data) {
    _data = data
    _rows = data.count
    _cols = data[0].count
  }

  // Create an empty matrix of specific dimensions
  static empty(rows, cols) {
    var data = List.filled(rows, null)
    for (i in 0...rows) data[i] = List.filled(cols, 0)
    return Matrix.new(data)
  }

  // Create an Identity matrix
  static identity(n) {
    var m = Matrix.empty(n, n)
    for (i in 0...n) m[i, i] = 1
    return m
  }

  static ones(r, c) {
    var m = Matrix.empty(r, c)
    for (i in 0...r) {
        for (j in 0...c) m[i, j] = 1
    }
    return m
  }

  rows { _rows }
  cols { _cols }
  data { _data }

  // Overload getters/setters for easy access: matrix[r, c]
  [r, c] { _data[r][c] }
  [r, c]=(val) { _data[r][c] = val }

  // --- Basic Operations ---
  add(other) {
    var m = Matrix.empty(_rows, _cols)
    for (r in 0..._rows) {
      for (c in 0..._cols) m[r, c] = this[r, c] + other[r, c]
    }
    return m
  }

  sub(other) {
    var m = Matrix.empty(_rows, _cols)
    for (r in 0..._rows) {
      for (c in 0..._cols) m[r, c] = this[r, c] - other[r, c]
    }
    return m
  }

  // Hadamard Product (Element-wise multiplication)
  times(other) {
    var m = Matrix.empty(_rows, _cols)
    for (r in 0..._rows) {
      for (c in 0..._cols) m[r, c] = this[r, c] * other[r, c]
    }
    return m
  }

  div(other) {
    var m = Matrix.empty(_rows, _cols)
    for (r in 0..._rows) {
      for (c in 0..._cols) {
        if (other[r, c] == 0) Fiber.abort("Division by zero in matrix.")
        m[r, c] = this[r, c] / other[r, c]
      }
    }
    return m
  }
 // Apply a function to every element (useful for activations)
  map(fn) {
    var m = Matrix.empty(_rows, _cols)
    for (r in 0..._rows) {
      for (c in 0..._cols) m[r, c] = fn.call(this[r, c])
    }
    return m
  }

  transpose() {
    var m = Matrix.empty(_cols, _rows)
    for (r in 0..._rows) {
      for (c in 0..._cols) {
        m[c, r] = this[r, c]
      }
    }
    return m
  }

  matmul(other) {
    if (_cols != other.rows) Fiber.abort("Incompatible dimensions for matmul.")
    var m = Matrix.empty(_rows, other.cols)
    for (i in 0..._rows) {
      for (j in 0...other.cols) {
        var sum = 0
        for (k in 0..._cols) {
          sum = sum + this[i, k] * other[k, j]
        }
        m[i, j] = sum
      }
    }
    return m
  }

  // --- LU Factorization (Doolittle Algorithm) ---
  // Returns a map { "L": Matrix, "U": Matrix }
  lu() {
    if (_rows != _cols) Fiber.abort("LU requires a square matrix.")
    var n = _rows
    var L = Matrix.identity(n)
    var U = Matrix.empty(n, n)

    for (i in 0...n) {
      for (k in i...n) { // Upper
        var sum = 0
        for (j in 0...i) sum = sum + (L[i, j] * U[j, k])
        U[i, k] = this[i, k] - sum
      }
      for (k in i+1...n) { // Lower
        var sum = 0
        for (j in 0...i) sum = sum + (L[k, j] * U[j, i])
        L[k, i] = (this[k, i] - sum) / U[i, i]
      }
    }
    return { "L": L, "U": U }
  }

  // --- QR Factorization (Gram-Schmidt Process) ---
  // Returns a map { "Q": Matrix, "R": Matrix }
  qr() {
    var n = _rows
    var m = _cols
    var Q = Matrix.empty(n, m)
    var R = Matrix.empty(m, m)

    for (j in 0...m) {
      var v = List.filled(n, 0)
      for (i in 0...n) v[i] = this[i, j] // Column j

      for (i in 0...j) {
        R[i, j] = 0
        for (k in 0...n) R[i, j] = R[i, j] + Q[k, i] * this[k, j]
        for (k in 0...n) v[k] = v[k] - R[i, j] * Q[k, i]
      }

      var norm = 0
      for (i in 0...n) norm = norm + v[i] * v[i]
      norm = norm.sqrt
      R[j, j] = norm

      for (i in 0...n) Q[i, j] = v[i] / norm
    }
    return { "Q": Q, "R": R }
  }

  // Inverse using LU decomposition
  inverse() {
    if (_rows != _cols) Fiber.abort("Only square matrices have inverses.")
    var n = _rows
    var factors = lu()
    var L = factors["L"]
    var U = factors["U"]
    var inv = Matrix.empty(n, n)

    // Solve AX = I column by column
    for (i in 0...n) {
      var b = List.filled(n, 0)
      b[i] = 1 // Identity column

      // Forward substitution: Ly = b
      var y = List.filled(n, 0)
      for (j in 0...n) {
        var sum = 0
        for (k in 0...j) sum = sum + L[j, k] * y[k]
        y[j] = b[j] - sum
      }

      // Backward substitution: Ux = y
      for (j in n-1..0) {
        var sum = 0
        for (k in j+1...n) sum = sum + U[j, k] * inv[k, i]
        inv[j, i] = (y[j] - sum) / U[j, j]
      }
    }
    return inv
  }

  // Determinant calculated via LU Decomposition
  // det(A) = det(L) * det(U). Since L has 1s on the diagonal, det(L) = 1.
  // Therefore, det(A) is simply the product of the diagonal of U.
  determinant() {
    if (_rows != _cols) Fiber.abort("Determinant requires a square matrix.")
    var factors = this.lu()
    var U = factors["U"]
    var det = 1
    for (i in 0..._rows) {
      det = det * U[i, i]
    }
    return det
  }

  // QR Algorithm to find all eigenvalues (Approximation)
  eigenvalues() {
    if (_rows != _cols) Fiber.abort("Eigenvalues require a square matrix.")
    var cur = this
    // 100 iterations is usually enough for convergence
    for (i in 0...100) {
      var res = cur.qr()
      // The magic of the QR algorithm: A_{k+1} = R_k * Q_k
      cur = res["R"].matmul(res["Q"])
    }

    // After many iterations, the matrix becomes (mostly) upper triangular.
    // The eigenvalues are the diagonal elements.
    var evals = []
    for (i in 0..._rows) {
      evals.add(cur[i, i])
    }
    return evals
  }

  // Power Iteration to find the dominant Eigenvector (Approximation)
  // This finds the vector corresponding to the eigenvalue with the largest magnitude.
  dominantEigenvector() {
    if (_rows != _cols) Fiber.abort("Eigenvectors require a square matrix.")

    // Start with a random guess vector [1, 1, ... 1]
    var v = List.filled(_rows, 1.0)

    for (iter in 0...50) {
      // Multiply: w = A * v
      var nextV = List.filled(_rows, 0.0)
      for (i in 0..._rows) {
        for (j in 0..._rows) {
          nextV[i] = nextV[i] + this[i, j] * v[j]
        }
      }

      // Normalize the vector (Magnitude = 1)
      var norm = 0
      for (i in 0..._rows) norm = norm + nextV[i] * nextV[i]
      norm = norm.sqrt

      for (i in 0..._rows) v[i] = nextV[i] / norm
    }
    return v
  }

  toString {
    var out = ""
    for (r in 0..._rows) {
      var rowStr = ""
      for (c in 0..._cols) {
        var val = (this[r, c] * 1000).round / 1000 // Round to 3 decimals
        var s = val.toString

        // Manual padLeft: Add spaces until the string is 8 characters long
        while (s.count < 8) {
          s = " " + s
        }

        rowStr = rowStr + s + " "
      }
      out = out + "[ " + rowStr + " ]\n"
    }
    return out
  }
}

