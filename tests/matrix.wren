import "../data/Matrix" for Matrix

var A = Matrix.new([
  [1, 2, 4],
  [3, 8, 14],
  [2, 6, 13]
])

System.print("Matrix A:")
System.print(A)

System.print("Transpose:")
System.print(A.transpose())

System.print("Inverse of A:")
System.print(A.inverse())

var decomp = A.lu()
System.print("LU Decomposition (L):")
System.print(decomp["L"])
System.print("LU Decomposition (U):")
System.print(decomp["U"])

decomp = A.qr()
System.print("QR Decomposition (Q):")
System.print(decomp["Q"])
System.print("QR Decomposition (R):")
System.print(decomp["R"])

System.print(A.eigenvalues())

System.print(A.dominantEigenvector())


