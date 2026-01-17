import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 1. Configurar Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Cargar y Congelar
model = models.resnet50(weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = False

# 3. Reemplazar Cabeza (La nueva capa sí tiene requires_grad=True por defecto)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) 

model = model.to(device)
model.train() # <--- Vital para BatchNormalization

# 4. Optimizador y Pérdida
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 5. Entrenamiento
for epoch in range(5):
    for inputs, labels in dataloader:
        # Mover datos al dispositivo
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
