import "random" for Random
import "../data/Matrix" for Matrix

class DataLoader {
  construct new(features, targets, batchSize, shuffle) {
    _features = features
    _targets = targets
    _batchSize = batchSize
    _shuffle = shuffle
    _rng = Random.new()
    
    _indices = []
    for (i in 0..._features.rows) _indices.add(i)
    
    // Calculate total number of batches
    _totalBatches = (_features.rows / _batchSize).ceil
  }

  // Shuffles the index mapping so we don't change the original matrices
  shuffle() {
    if (!_shuffle) return
    for (i in _indices.count - 1..1) {
      var j = _rng.int(i + 1)
      var temp = _indices[i]
      _indices[i] = _indices[j]
      _indices[j] = temp
    }
  }

  // Returns a list of batches, where each batch is a map { "x": Matrix, "y": Matrix }
  getBatches() {
    shuffle()
    var batches = []

    for (b in 0..._totalBatches) {
      var start = b * _batchSize
      var end = (start + _batchSize < _features.rows) ? start + _batchSize : _features.rows
      var actualBatchSize = end - start
      
      var batchX = Matrix.empty(actualBatchSize, _features.cols)
      var batchY = Matrix.empty(actualBatchSize, _targets.cols)

      for (i in 0...actualBatchSize) {
        var originalIndex = _indices[start + i]
        
        // Copy Features
        for (c in 0..._features.cols) {
          batchX[i, c] = _features[originalIndex, c]
        }
        
        // Copy Targets
        for (c in 0..._targets.cols) {
          batchY[i, c] = _targets[originalIndex, c]
        }
      }
      
      batches.add({ "x": batchX, "y": batchY })
    }
    
    return batches
  }

  count { _totalBatches }
}
