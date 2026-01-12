import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

def ejecutar_demo_cnn():
    # 1. CARGAR IMAGEN (Con truco para evitar bloqueos)
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/600px-Cat03.jpg"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        print("Descargando imagen...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('L') # Escala de grises
    except Exception as e:
        print(f"No se pudo descargar la imagen ({e}). Creando una imagen de prueba...")
        # Crear imagen de prueba (cuadrado blanco en fondo negro)
        img_np = np.zeros((200, 200), dtype=np.uint8)
        img_np[50:150, 50:150] = 255
        img = Image.fromarray(img_np)

    # Convertir a Tensor de PyTorch (Batch, Canales, Alto, Ancho)
    img_tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0)

    # 2. DEFINIR EL FILTRO (Detector de bordes verticales - Sobel)
    # Este filtro resalta cambios de intensidad de izquierda a derecha
    kernel = torch.tensor([[[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]]], dtype=torch.float32)

    # --- PROCESAMIENTO PASO A PASO ---

    # A. PADDING: Añadimos 10 píxeles de borde negro
    # El padding se aplica como (izq, der, arriba, abajo)
    img_padded = F.pad(img_tensor, (10, 10, 10, 10), mode='constant', value=0)

    # B. CONVOLUCIÓN: Aplicamos el filtro
    conv_out = F.conv2d(img_padded, kernel)

    # C. ACTIVACIÓN (ReLU): Matamos los valores negativos
    relu_out = F.relu(conv_out)

    # D. MAX POOLING: Reducimos a la mitad (2x2)
    # Se queda con el valor más alto de cada bloque de 2x2
    pool_out = F.max_pool2d(relu_out, kernel_size=2, stride=2)

    # E. FLATTEN: Convertimos la imagen en un vector (lista larga de números)
    vector_final = pool_out.view(-1).detach().numpy()

    # --- VISUALIZACIÓN ---
    print("Generando visualización...")
    fig, axs = plt.subplots(1, 5, figsize=(22, 5))
    
    # 1. Original
    axs[0].imshow(np.array(img), cmap='gray')
    axs[0].set_title("1. Entrada\nOriginal")
    axs[0].axis('off')

    # 2. Padding
    axs[1].imshow(img_padded.squeeze(), cmap='gray')
    axs[1].set_title("2. Padding\n(Borde de ceros)")
    axs[1].axis('off')

    # 3. Convolución
    axs[2].imshow(conv_out.squeeze().detach().numpy(), cmap='gray')
    axs[2].set_title("3. Convolución\n(Filtro de bordes)")
    axs[2].axis('off')

    # 4. ReLU
    axs[3].imshow(relu_out.squeeze().detach().numpy(), cmap='gray')
    axs[3].set_title("4. ReLU\n(Solo positivos)")
    axs[3].axis('off')

    # 5. Pooling
    axs[4].imshow(pool_out.squeeze().detach().numpy(), cmap='gray')
    axs[4].set_title(f"5. Max Pooling\n(Resumen {pool_out.shape[2]}x{pool_out.shape[3]})")
    axs[4].axis('off')

    plt.tight_layout()
    plt.show()

    # Visualización del Vector (Flatten)
    plt.figure(figsize=(12, 3))
    plt.plot(vector_final[:1000]) # Mostramos solo los primeros 1000 valores para que se vea algo
    plt.title(f"6. Vectorización (Flatten) - Tamaño total del vector: {len(vector_final)}")
    plt.xlabel("Índice de la neurona")
    plt.ylabel("Activación")
    plt.grid(alpha=0.3)
    plt.show()

    print(f"Proceso completado.")
    print(f"Dimensiones finales: {pool_out.shape}")

# Ejecutar la función
ejecutar_demo_cnn()
