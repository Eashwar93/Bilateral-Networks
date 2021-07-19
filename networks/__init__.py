

from .bisenetv1 import BiSeNetV1
from .bisenet_v1_g1 import BiSeNetV1_g1
from .bisenet_v1_g2 import BiSeNetV1_g2
from .bisenet_v1_g3 import BiSeNetV1_g3
from .bisenet_v1_g4 import BiSeNetV1_g4
from .bisenet_v1_g5 import BiSeNetV1_g5
from .bisenet_v1_g6 import BiSeNetV1_g6
from .bisenet_v1_g7 import BiSeNetV1_g7
from .bisenet_v1_g8 import BiSeNetV1_g8
from .fanet18_v1 import FANet18_v1
from .fanet18_v2 import FANet18_v2
from .fanet18_v3 import FANet18_v3
from .fanet18_v4 import FANet18_v4
from .fanet18_v1_se1 import FANet18_v1_se1
from .fanet18_v1_c1 import FANet18_v1_c1
from .fanet18_v1_se2 import FANet18_v1_se2
from .fanet18_v1_se3 import FANet18_v1_se3
from .fanet18_v4_se1 import FANet18_v4_se1
from .fanet18_v4_se2 import FANet18_v4_se2
from .fanet18_v4_se2_c1 import FANet18_v4_se2_c1
from .fanet18_v4_se2_c2 import FANet18_v4_se2_c2


model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenet_v1_g1': BiSeNetV1_g1,
    'bisenet_v1_g2': BiSeNetV1_g2,
    'bisenet_v1_g3': BiSeNetV1_g3,
    'bisenet_v1_g4': BiSeNetV1_g4,
    'bisenet_v1_g5': BiSeNetV1_g5,
    'bisenet_v1_g6':BiSeNetV1_g6,
    'bisenet_v1_g7': BiSeNetV1_g7,
    'bisenet_v1_g8': BiSeNetV1_g8,
    'fanet18_v1': FANet18_v1,
    'fanet18_v2': FANet18_v2,
    'fanet18_v3': FANet18_v3,
    'fanet18_v4': FANet18_v4,
    'fanet18_v1_se1': FANet18_v1_se1,
    'fanet18_v1_c1': FANet18_v1_c1,
    'fanet18_v1_se2': FANet18_v1_se2,
    'fanet18_v1_se3': FANet18_v1_se3,
    'fanet18_v4_se1': FANet18_v4_se1,
    'fanet18_v4_se2': FANet18_v4_se2,
    'fanet18_v4_se2_c1': FANet18_v4_se2_c1,
    'fanet18_v4_se2_c2': FANet18_v4_se2_c2,
}