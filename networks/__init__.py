

from .bisenetv1 import BiSeNetV1
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


model_factory = {
    'bisenetv1': BiSeNetV1,
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
}