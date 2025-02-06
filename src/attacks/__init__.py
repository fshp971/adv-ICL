from .gcg import GCG
from .beast import BEAST


__attacker_zoo__ = {
    "gcg"   : GCG,
    "beast" : BEAST,
}

def build_attacker(name: str, **kwargs):
    return __attacker_zoo__[name](**kwargs)
