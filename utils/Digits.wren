import "../data/Matrix" for Matrix
import "io" for File

class Digits {
  construct new() {
    _rawFeatures = []
    _rawLabels = []
  }

  // Reads the CSV and parses strings to numbers
  load(path) {
    var content = File.read(path)
    var lines = content.split("\n")
    for (line in lines) {
      var trimmed = line.trim()
      if (trimmed.count == 0) continue
      
      var parts = trimmed.split(",")
      var features = []
      for (i in 0...64) {
        features.add(Num.fromString(parts[i].trim()) / 16.0)
      }
      _rawFeatures.add(features)
      _rawLabels.add(Num.fromString(parts[64].trim()))
    }
    System.print("Loaded %(_rawLabels.count) samples.")
  }

  // Returns N x 16 Matrix
  features { Matrix.new(_rawFeatures) }

  // Returns N x 10 Matrix (One-hot encoding)
  targets {
    var n = _rawLabels.count
    var m = Matrix.empty(n, 10)
    for (i in 0...n) {
      var label = _rawLabels[i].round
      if (label >= 0 && label < 10) {
        m[i, label] = 1
      }
    }
    return m
  }

  // Visualizes the 4x4 image in the console
  showImage(index) {
    if (index < 0 || index >= _rawFeatures.count) {
      System.print("Index out of bounds.")
      return
    }
    
    var pixels = _rawFeatures[index]
    var label = _rawLabels[index]
    
    System.print("\nDigit Label: %(label) (Index: %(index))")
    System.print("+----------------+")
    for (y in 0...8) {
      var rowString = "|"
      for (x in 0...8) {
        var val = pixels[y * 8 + x]
        // Use shaded blocks based on pixel intensity (0-100)
        if (val > 0.75) {
          rowString = rowString + "██" 
        } else if (val > 0.5) {
          rowString = rowString + "▒▒"
        } else if (val > 0.25) {
          rowString = rowString + "░░"
        } else {
          rowString = rowString + "  "
        }
      }
      System.print(rowString + "|")
    }
    System.print("+----------------+")
  }
}
