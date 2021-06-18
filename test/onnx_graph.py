import onnx

model = onnx.load("./res/monorail_model.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)