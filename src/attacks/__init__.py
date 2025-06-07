from .gcg import GCG
from .beast import BEAST
from .autodan_zhu import AutoDANZhu
from .conti_embed import ContiEmbed


__attacker_zoo__ = {
    "gcg"   : GCG,
    "beast" : BEAST,
    "autodan-zhu" : AutoDANZhu,
    "conti-embed" : ContiEmbed,
}

def build_attacker(name: str, **kwargs):
    return __attacker_zoo__[name](**kwargs)
