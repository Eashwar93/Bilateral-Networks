from .bisenetv1 import cfg as bisenetv1_cfg
from .bisenet_v1_g1 import cfg as bisenet_v1_g1_cfg
from .bisenet_v1_g2 import cfg as bisenet_v1_g2_cfg
from .bisenet_v1_g3 import cfg as bisenet_v1_g3_cfg
from .bisenet_v1_g4 import cfg as bisenet_v1_g4_cfg
from .bisenet_v1_g5 import cfg as bisenet_v1_g5_cfg
from .bisenet_v1_g6 import cfg as bisenet_v1_g6_cfg
from .bisenet_v1_g7 import cfg as bisenet_v1_g7_cfg
from .bisenetv2 import cfg as bisenetv2_cfg
from .fanet18_v1 import cfg as fanet18_v1_cfg
from .fanet18_v2 import cfg as fanet18_v2_cfg
from .fanet18_v3 import cfg as fanet18_v3_cfg
from .fanet18_v4 import cfg as fanet18_v4_cfg
from .fanet18_v1_se1 import cfg as fanet18_v1_se1_cfg
from .fanet18_v1_c1 import cfg as fanet18_v1_c1_cfg
from .fanet18_v1_se2 import cfg as fanet18_v1_se2_cfg
from .fanet18_v1_se3 import cfg as fanet18_v1_se3_cfg
from .fanet18_v4_se1 import cfg as fanet18_v4_se1_cfg
from .fanet18_v4_se2 import cfg as fanet18_v4_se2_cfg
from .fanet18_v4_se2_c1 import cfg as fanet18_v4_se2_c1_cfg
from .fanet18_v4_se2_c2 import cfg as fanet18_v4_se2_c2_cfg

class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d

cfg_factory = dict(
    bisenetv1=cfg_dict(bisenetv1_cfg),
    bisenet_v1_g1=cfg_dict(bisenet_v1_g1_cfg),
    bisenet_v1_g2=cfg_dict(bisenet_v1_g2_cfg),
    bisenet_v1_g3=cfg_dict(bisenet_v1_g3_cfg),
    bisenet_v1_g4=cfg_dict(bisenet_v1_g4_cfg),
    bisenet_v1_g5=cfg_dict(bisenet_v1_g5_cfg),
    bisenet_v1_g6=cfg_dict(bisenet_v1_g6_cfg),
    bisenet_v1_g7=cfg_dict(bisenet_v1_g7_cfg),
    bisenetv2=cfg_dict(bisenetv2_cfg),
    fanet18_v1=cfg_dict(fanet18_v1_cfg),
    fanet18_v2=cfg_dict(fanet18_v2_cfg),
    fanet18_v3=cfg_dict(fanet18_v3_cfg),
    fanet18_v4=cfg_dict(fanet18_v4_cfg),
    fanet18_v1_se1=cfg_dict(fanet18_v1_se1_cfg),
    fanet18_v1_c1=cfg_dict(fanet18_v1_c1_cfg),
    fanet18_v1_se2=cfg_dict(fanet18_v1_se2_cfg),
    fanet18_v1_se3=cfg_dict(fanet18_v1_se3_cfg),
    fanet18_v4_se1=cfg_dict(fanet18_v4_se1_cfg),
    fanet18_v4_se2=cfg_dict(fanet18_v4_se2_cfg),
    fanet18_v4_se2_c1=cfg_dict(fanet18_v4_se2_c1_cfg),
    fanet18_v4_se2_c2=cfg_dict(fanet18_v4_se2_c2_cfg),
)