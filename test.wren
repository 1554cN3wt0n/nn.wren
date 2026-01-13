import "./data/Matrix" for Matrix
import "./data/Variable" for Variable

var m1 = Variable.new(Matrix.new([[2,3],[4,5]]))
var m2 = Variable.new(Matrix.new([[2,3]]))
var c = Variable.new(Matrix.new([[1],[1]]))

System.print(Variable.add(m1,Variable.matmul(c,m2)).data)

