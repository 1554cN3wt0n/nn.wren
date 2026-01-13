import "../utils/Digits" for Digits

// --- Main Execution ---
var loader = Digits.new()
loader.load("./resources/digits.csv")

// 1. Get the Matrices
var X = loader.features
var Y = loader.targets

System.print("\nFeature Matrix (First 2 rows, 16 columns):")
// Showing only a slice for brevity
for (i in 0...2) System.print(X.data[i])

System.print("\nOne-Hot Target Matrix (First 5 rows, 10 columns):")
for (i in 0...5) System.print(Y.data[i])

// 2. Show some images
loader.showImage(0)
loader.showImage(100)
loader.showImage(200)
