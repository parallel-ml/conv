from ..models import c3d
from ..timer import layerwise
import sys


def test_c3d():
    path = 'timer/resource/c3d/c3d'

    sys.stdout = open(path + '.txt', 'w+')
    model = c3d.original()
    print layerwise.timer(model)
