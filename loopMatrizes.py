import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# MODELO
class FCN_Apresentacao1D(nn.Module):

    def __init__(self, input_size=512):
        super().__init__()

        self.features = nn.Sequential(

            #reduzidos para 8, ReLU e Dropout 0.1
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),

            #reduzidos para 16, ReLU e Dropout 0.1
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),

            #reduzidos para 32, ReLU
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # receber o tamanho de saída do último filtro (32)
        self.classifier = nn.Conv1d(32, 2, kernel_size=1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = self.classifier(x)

        return x.view(x.size(0), -1)


# SALVAMENTO
DIRETORIO_SALVAMENTO = '/content/drive/Shareddrives/Projeto_TEA/Dados_TEA_V2/resultados'
os.makedirs(DIRETORIO_SALVAMENTO, exist_ok=True)

caminho_modelo_final = os.path.join(
    DIRETORIO_SALVAMENTO,
    'melhor_modelo1d.pth'
)


# CONFIG
model = FCN_Apresentacao1D().to(DEVICE)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-3
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2
)


# CLASS IMBALANCE
count_0 = (tensor_y_train == 0).sum().item()
count_1 = (tensor_y_train == 1).sum().item()

print(f"\nDistribuicao no Treino -> TC (0): {count_0} | TEA (1): {count_1}")

total = count_0 + count_1

peso_0 = total / (2 * count_0)
peso_1 = total / (2 * count_1)

pesos = torch.tensor([peso_0, peso_1]).float().to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=pesos)



# TREINO
num_epochs = 100
best_auc = 0.0
early_break = 10
epochs_sem_melhora = 0

history = {
    'train_loss': [],
    'val_loss': [],
    'val_auc': []
}

print("Iniciando treinamento da FCN 1D (Avaliacao por AUC)...\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step(epoch)


    # VALIDACAO
    model.eval()
    val_loss = 0.0
    all_true_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            v_loss = criterion(outputs, labels)
            val_loss += v_loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    try:
        auc = roc_auc_score(all_true_labels, all_probs) * 100
    except ValueError:
        auc = 0.0

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_auc'].append(auc)

  
    # EARLY STOPPING
    if auc > best_auc:
        best_auc = auc
        epochs_sem_melhora = 0
        torch.save(
            model.state_dict(),
            caminho_modelo_final
        )
        print(
            f"Ep {epoch+1:02d}: "
            f"T-Loss {avg_train_loss:.4f} | "
            f"V-Loss {avg_val_loss:.4f} | "
            f"AUC {auc:.1f}% (Salvo!)"
        )

    else:
        epochs_sem_melhora += 1
        print(
            f"Ep {epoch+1:02d}: "
            f"T-Loss {avg_train_loss:.4f} | "
            f"V-Loss {avg_val_loss:.4f} | "
            f"AUC {auc:.1f}%"
        )

        if epochs_sem_melhora >= early_break:
            print(
                f"\n=> Early Stopping ativado na epoca {epoch+1}! "
                f"O modelo parou de melhorar a AUC."
            )
            break

# RESULTADOS
preds_tensor = torch.tensor(all_preds)
unique_preds, counts = torch.unique(
    preds_tensor,
    return_counts=True
)

chutes = dict(zip(
    unique_preds.tolist(),
    counts.tolist()
))

print(f"\nUltima Epoca - Predicoes na Validacao: {chutes}")
print(
    f"Melhor AUC Alcançada: {best_auc:.1f}% "
    f"(Modelo salvo em {caminho_modelo_final})"
)



# GRAFICOS
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Treino')
plt.plot(history['val_loss'], label='Validacao')
plt.title('Curva de Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history['val_auc'], color='purple')
plt.title('Area Sob a Curva ROC (AUC)')
plt.show()