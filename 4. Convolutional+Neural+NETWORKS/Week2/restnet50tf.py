import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# 1. Cargar base y congelar
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# 2. Construir modelo con Capa de Preprocesamiento
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Lambda(preprocess_input), # <--- VITAL: Ajusta los colores para ResNet
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),             # Opcional: evita sobreajuste
    layers.Dense(10, activation='softmax') 
])

# 3. Preparar Datos
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train', 
    image_size=(224, 224),
    batch_size=32
)

# 4. Compilar y Entrenar
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(train_ds, epochs=5)
