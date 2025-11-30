"""
Script para comparar si walkforward_permutation.py ahora da los mismos
resultados que walkforward_donchian_mod.py (sin seed)
"""

print("""
IMPORTANTE: Para que los resultados sean IDÉNTICOS, ambos códigos deben:

1. NO usar seed (para tener permutaciones completamente aleatorias)
   O
2. Ambos usar el MISMO seed con el MISMO orden de generación

Como walkforward_donchian_mod.py NO usa seed, he quitado el seed
de walkforward_permutation.py temporalmente.

Ejecuta:
  python walkforward_donchian_mod.py donchian.py
  python walkforward_permutation.py donchian

Y compara los resultados. Deberían estar en el mismo rango estadístico
(aunque no idénticos por ser aleatorios).

Si quieres resultados REPRODUCIBLES, debes agregar seed a AMBOS códigos.
""")
