---
# My Diary

Today  **{{DATE_PRETTY}}**.

## General Counter
It has been    **{{DAY_SINCE_2025_08_18}}** days since I started this diary.

# Repository Counter

Day **{{DAYS_SINCE_REPO_START}}** From I started this repository


###  Pyhon
```

import numpy as np

# 1. Define nx (ejemplo: 12288 para imágenes de 64x64x3)
nx = 12288 

# 2. Define la estructura de la red según tu examen
layer_dims = [nx, 4, 3, 2, 1] 

# 3. Inicializa el diccionario de parámetros
parameter = {}

# 4. Tu código ahora correrá sin errores:
for i in range(1, len(layer_dims)):
    parameter['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
    parameter['b' + str(i)] = np.random.randn(layer_dims[i], 1) * 0.01
print(parameter)
```
