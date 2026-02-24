import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score 

# arquitetura fcn 1d para os vetores
class FCN_Apresentacao1D(nn.Module):
    def __init__(self):
        super(FCN_Apresentacao1D, self).__init__()
        
       
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Camada final de classificacao
        self.classifier = nn.Conv1d(64, 2, kernel_size=1)

    def forward(self, x):
        # Garante que a entrada tenha a dimensao de canal (Batch, Canal=1, Features=256)
        if x.dim() == 2:
            x = x.unsqueeze(1) 
            
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

# caminhos
DIRETORIO_SALVAMENTO = '/content/drive/Shareddrives/Projeto_TEA/Dados_TEA_V2/resultados'
os.makedirs(DIRETORIO_SALVAMENTO, exist_ok=True)
caminho_modelo_final = os.path.join(DIRETORIO_SALVAMENTO, 'melhor_modelo1d.pth')

# config modelo, otimizador, scheduler e criterio de perda
model = FCN_Apresentacao1D().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Calcula os pesos para a perda com base na distribuicao das classes no treino
count_0 = (tensor_y_train == 0).sum().item()
count_1 = (tensor_y_train == 1).sum().item()
print(f"\nDistribuicao no Treino -> TC (0): {count_0} | TEA (1): {count_1}")

pesos = torch.tensor([1.0 / count_0, 1.0 / count_1], dtype=torch.float).to(DEVICE)
pesos = pesos / pesos.sum()
criterion = nn.CrossEntropyLoss(weight=pesos)

num_epochs = 100 
best_auc = 0.0
early_break = 10
epochs_sem_melhora = 0
history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

print("Iniciando treinamento da FCN 1D (Avaliacao por AUC)...\n")

# loop de treino
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
        
    # --- VALIDACAO ---
    model.eval()
    val_loss = 0.0
    all_true_labels = []
    all_probs = [] # Para a AUC, precisamos das probabilidades, nao apenas dos chutes
    all_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            v_loss = criterion(outputs, labels)
            val_loss += v_loss.item()
            
            # Aplica Softmax para pegar a probabilidade da Classe 1 (TEA)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            
            all_probs.extend(probs.tolist())
            all_true_labels.extend(labels.tolist())
            all_preds.extend(predicted.tolist())
            
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    # Calcula a metrica AUC
    try:
        auc = roc_auc_score(all_true_labels, all_probs) * 100
    except ValueError:
        auc = 0.0 # Previne erro se o batch tiver apenas uma classe
        
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_auc'].append(auc)
    scheduler.step()
    
    # Baseado na AUC, decide se salva o modelo e se ativa o early stopping
    if auc > best_auc:
        best_auc = auc
        epochs_sem_melhora = 0
        torch.save(model.state_dict(), caminho_modelo_final)
        print(f"Ep {epoch+1:02d}: T-Loss {avg_train_loss:.4f} | V-Loss {avg_val_loss:.4f} | AUC {auc:.1f}% (Salvo!)")
    else:
        epochs_sem_melhora += 1
        print(f"Ep {epoch+1:02d}: T-Loss {avg_train_loss:.4f} | V-Loss {avg_val_loss:.4f} | AUC {auc:.1f}%")
        
        if epochs_sem_melhora >= early_break:
            print(f"\n=> Early Stopping ativado na epoca {epoch+1}! O modelo parou de melhorar a AUC.")
            break

# --- 5. DIAGNOSTICO FINAL E GRAFICOS ---
preds_tensor = torch.tensor(all_preds)
unique_preds, counts = torch.unique(preds_tensor, return_counts=True)
chutes = dict(zip(unique_preds.tolist(), counts.tolist()))

print(f"\nUltima Epoca - Predicoes na Validacao: {chutes}")
print(f"Melhor AUC Alcançada: {best_auc:.1f}% (Modelo salvo em {caminho_modelo_final})")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Treino')
plt.plot(history['val_loss'], label='Validacao')
plt.title('Curva de Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_auc'], color='purple')
plt.title('Area Sob a Curva ROC (AUC)')
plt.show()