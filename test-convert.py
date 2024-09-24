from convert.convert_to_ctranslate2 import convert_to_ctranslate2
from sympy.physics.units import pa


path = "models/large-v3-big-dataset"

convert_to_ctranslate2(path, path + '-ct2')