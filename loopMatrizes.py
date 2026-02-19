import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Extrai as labels da lista train_subj gerada no Codigo 2
labels_train = [s['label'] for s in train_subj]

#ARQUITETURA FCN SIMPLIFICADA ---
class FCN_Apresentacao(nn.Module):
    def __init__(self):
        super(FCN_Apresentacao, self).__init__()
        
        self.features = nn.Sequential(#Camadas convolucionais para extrair características das matrizes de conectividade. O uso de BatchNorm e Dropout ajuda a estabilizar o treinamento e reduzir o overfitting.
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Camada final de classificação (1x1 conv para manter a estrutura de FCN, seguida de flatten)
        self.classifier = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)




DIRETORIO_SALVAMENTO = '/content/drive/Shareddrives/Projeto_TEA/Dados_TEA_V2/resultados'
os.makedirs(DIRETORIO_SALVAMENTO, exist_ok=True)
caminho_modelo_final = os.path.join(DIRETORIO_SALVAMENTO, 'melhor_modelo_fcn.pth')

# --- 4. CONFIGURACAO ---
model = FCN_Apresentacao().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)#Essa parte é crucial para evitar overfitting, especialmente com um dataset pequeno. O weight_decay penaliza pesos muito grandes, o que ajuda a generalizar melhor. O learning rate de 1e-4 é um bom ponto de partida para uma FCN, mas pode ser ajustado conforme necessário.    
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Pesos para balancear as classes
count_0 = labels_train.count(0)
count_1 = labels_train.count(1)
print(f"\nDistribuicao no Treino -> TC (0): {count_0} | TEA (1): {count_1}")
pesos = torch.tensor([1.0 / count_0, 1.0 / count_1], dtype=torch.float).to(DEVICE)
pesos = pesos / pesos.sum()
criterion = nn.CrossEntropyLoss(weight=pesos)#

num_epochs = 60
best_acc = 0.0
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

print("Iniciando treinamento da FCN para Apresentacao...\n")

# --- 5. LOOP DE TREINAMENTO ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # Validacao
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    all_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            v_loss = criterion(outputs, labels)
            
            val_loss += v_loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.tolist())
            
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    acc = 100 * correct / total
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(acc)
    scheduler.step()
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), caminho_modelo_final)
        print(f"Ep {epoch+1:02d}: T-Loss {avg_train_loss:.4f} | V-Loss {avg_val_loss:.4f} | Acc {acc:.1f}% (Salvo!)")
    else:
        print(f"Ep {epoch+1:02d}: T-Loss {avg_train_loss:.4f} | V-Loss {avg_val_loss:.4f} | Acc {acc:.1f}%")





# Diagnóstico final e gráficos
preds_tensor = torch.tensor(all_preds)
unique_preds, counts = torch.unique(preds_tensor, return_counts=True)
chutes = dict(zip(unique_preds.tolist(), counts.tolist()))
print(f"\nUltima Epoca - Predicoes na Validacao: {chutes}")
print(f"Melhor Acuracia: {best_acc:.1f}% (Salvo em {caminho_modelo_final})")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Treino')
plt.plot(history['val_loss'], label='Validacao')
plt.title('Curva de Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], color='green')
plt.title('Acuracia de Validacao')
plt.show()