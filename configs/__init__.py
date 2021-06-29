from .bisenetv1 import cfg as bisenetv1_cfg
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