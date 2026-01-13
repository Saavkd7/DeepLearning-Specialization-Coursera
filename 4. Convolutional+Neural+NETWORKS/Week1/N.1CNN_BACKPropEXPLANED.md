# Guía Completa de Redes Neuronales Convolucionales (CNN)

Esta guía explica desde cero cómo funcionan las CNN, usando la analogía de detectives y piezas de Lego, e incluye las implementaciones en Python con NumPy.

---

## 1. Conceptos Fundamentales (Explicación para 12 años)

### ¿Qué es una CNN?
Es como un **equipo de detectives súper especializados** que intentan adivinar qué hay en una foto. No ven la imagen de golpe, la analizan por pedacitos.

### El Detective con la Lupa (El Filtro)
Una computadora solo ve números. El detective usa una "lupa" (un filtro de $3 \times 3$ números) que va saltando por toda la imagen.
* **La Multiplicación:** Compara sus números con los de la imagen. Si coinciden, el resultado es un número alto.
* **La Gran Suma:** Suma todos esos resultados para dar un "Marcador de Éxito". Si el marcador es alto, el detective grita: "¡Aquí hay un borde!".

### El Volumen (El pastel de capas)
A partir de la segunda capa, los detectives miran un **volumen**. Si la capa anterior detectó bordes verticales en una capa y horizontales en otra, el detective actual mira ambas al mismo tiempo.
* **Matemática:** $Vertical + Horizontal = Esquina$. Así es como se forman objetos complejos.

---

## 2. Implementación de Funciones Core

### A. Zero Padding (Acolchado)
Sirve para no perder información en los bordes y mantener el tamaño de la imagen.

```python
def zero_pad(X, pad):
    """
    X -- (m, n_H, n_W, n_C)
    pad -- cantidad de ceros alrededor
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))
    return X_pad
