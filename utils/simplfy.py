import onnx
from onnxsim import simplify

model = onnx.load("./res/monorail_model_opset11.onnx")

model_sim, check = simplify(model)
onnx.save(model_sim, "./res/monorail_opset11_simpl.onnx")