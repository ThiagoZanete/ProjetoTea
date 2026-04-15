"""
Objetivo:
    - Definir a arquitetura FCN 1D para classificação de conectividade funcional.
    - Configurar otimizador, scheduler e função de perda com suporte a
      desbalanceamento de classes (Focal Loss).
    - Treinar o modelo com monitoramento por AUC-ROC e Early Stopping.
    - Avaliar o melhor modelo salvo no conjunto de Teste (holdout final).
 
Dependências do Módulo 1:
    - train_loader, val_loader, test_loader  : DataLoaders prontos para uso.
    - tensor_y_train                         : Rótulos do treino (para balanceamento).
    - DEVICE                                 : CPU ou CUDA definido no Módulo 1.
"""
 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
 
 
#   ARQUITETURA DA REDE (FCN 1D)
#
# Rede Convolucional 1D totalmente convolucional (FCN) que opera sobre o vetor
# de conectividade funcional (512 features) tratando-o como um sinal 1D.
#
# Motivação do design:
#   - Conv1d com stride=2 substitui MaxPool, reduzindo a dimensão e aprendendo
#     a subamostragem ao mesmo tempo.
#   - BatchNorm + Dropout após cada bloco convolucional controlam overfitting.
#   - AvgPool1d estático (kernel=64) no lugar de AdaptiveAvgPool: garante
#     comportamento determinístico e facilita futura portabilidade para FPGA.
#   - Classificador final com Conv1d(32, 2, kernel=1) em vez de Linear:
#     mantém a natureza totalmente convolucional da rede.
 
class FCN_Apresentacao1D(nn.Module):
    def __init__(self, input_size=512):
        super().__init__()
 
        self.features = nn.Sequential(
            # Bloco 1: extrai padrões locais de baixo nível
            # Entrada : (Batch,  1, 512)
            nn.Conv1d(1, 8, kernel_size=3, padding=1, stride=2),
            # Saída   : (Batch,  8, 256)
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout1d(0.2),
 
            # Bloco 2: combina padrões do bloco anterior em representações mais abstratas
            # Entrada : (Batch,  8, 256)
            nn.Conv1d(8, 16, kernel_size=3, padding=1, stride=2),
            # Saída   : (Batch, 16, 128)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout1d(0.3),
 
            # Bloco 3: comprime a representação para um vetor rico em semântica
            # Entrada : (Batch, 16, 128)
            nn.Conv1d(16, 32, kernel_size=3, padding=1, stride=2),
            # Saída   : (Batch, 32,  64)
            nn.BatchNorm1d(32),
            nn.ReLU(),
 
            # Pooling estático: colapsa a dimensão temporal para um único valor por canal
            # Entrada : (Batch, 32, 64)
            nn.AvgPool1d(kernel_size=64),
            # Saída   : (Batch, 32,  1)
        )
 
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            # Projeção linear final via Conv1d(kernel=1) — equivalente a nn.Linear(32, 2)
            # mas mantém a arquitetura totalmente convolucional
            nn.Conv1d(32, 2, kernel_size=1),
            # Saída: (Batch, 2, 1)
        )
 
    def forward(self, x):
        # Garante que vetores 2D (Batch, 512) ganhem o canal de entrada necessário
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (Batch, 512) → (Batch, 1, 512)
 
        x = self.features(x)
        x = self.classifier(x)
 
        return x.view(x.size(0), -1)  # (Batch, 2, 1) → (Batch, 2)
 

#   CONFIGURAÇÃO DE SALVAMENTO
DIRETORIO_SALVAMENTO = '/content/drive/Shareddrives/Projeto_TEA/Dados_TEA_V2/resultados'
os.makedirs(DIRETORIO_SALVAMENTO, exist_ok=True)
 
caminho_modelo_final = os.path.join(
    DIRETORIO_SALVAMENTO,
    'modeloTeste'
)
 
 

#   INSTANCIAÇÃO DO MODELO, OTIMIZADOR E SCHEDULER
model = FCN_Apresentacao1D().to(DEVICE)
 
# Adam com weight decay leve (L2) para regularização implícita dos pesos
optimizer = optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=5e-4
)
 
# ReduceLROnPlateau monitora a AUC de validação:
# se ela não melhorar por 'patience' épocas, o LR é multiplicado por 'factor'.
# Isso evita oscilações tardias sem congelar o aprendizado prematuramente.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',     # Maximizar a AUC (ao contrário da loss, que se minimiza)
    factor=0.5,     # Reduz o LR à metade a cada plateau
    patience=5,     # Aguarda 5 épocas sem melhora antes de cortar o LR
)
 
 
# SEÇÃO 4 — FUNÇÃO DE PERDA COM SUPORTE A DESBALANCEAMENTO (FOCAL LOSS)
# O ABIDE é desbalanceado: há mais controles (TC) do que pacientes com TEA.
# A Focal Loss resolve isso duplamente:
#   1. Reduz o peso de exemplos fáceis (bem classificados), focando nos difíceis.
#   2. O parâmetro gamma > 1 amplifica esse foco nos casos mais ambíguos.
#
# Os pesos por classe são calculados a partir da distribuição real do treino,
# mas ficam disponíveis como referência (usados no critério anterior, se reativado).
 
count_0 = (tensor_y_train == 0).sum().item()
count_1 = (tensor_y_train == 1).sum().item()
print(f"\nDistribuicao no Treino -> TC (0): {count_0} | TEA (1): {count_1}")
 
total  = count_0 + count_1
peso_0 = total / (2 * count_0)
peso_1 = total / (2 * count_1)
pesos  = torch.tensor([peso_0, peso_1]).float().to(DEVICE)
 
 
class FocalLoss(nn.Module):
    """
    Reformula a Cross-Entropy padrão para down-pesar exemplos fáceis
    (alta confiança) e focar o gradiente nos exemplos difíceis ou mal
    classificados — especialmente útil em datasets desbalanceados.
    Parâmetros:
        alpha     : fator de escala global da loss (padrão=1, sem escala).
        gamma     : grau de foco nos exemplos difíceis; gamma=0 equivale à CE padrão.
        reduction : 'mean' (padrão) ou 'sum' para agregar o batch.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction
 
    def forward(self, inputs, targets):
        # Calcula a Cross-Entropy sem redução para obter a perda por amostra
        ce_loss   = F.cross_entropy(inputs, targets, reduction='none')
        # p_t = probabilidade atribuída à classe correta (via exp da CE negada)
        pt         = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
 
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
 
 
criterion = FocalLoss(gamma=2.5).to(DEVICE)
 

#   LOOP DE TREINAMENTO COM EARLY STOPPING
# Métrica de referência: AUC-ROC (robusta ao desbalanceamento de classes).
# O modelo é salvo sempre que a AUC de validação supera o melhor valor anterior.
# Se não houver melhora por 'early_break' épocas consecutivas, o treino é
# interrompido para evitar overfitting.
 
num_epochs        = 100
best_auc          = 0.0
early_break       = 10
epochs_sem_melhora = 0
 
history = {
    'train_loss': [],
    'val_loss':   [],
    'val_auc':    []
}
 
print("Iniciando treinamento da FCN 1D (Avaliacao por AUC)...\n")
 
for epoch in range(num_epochs):
 
    # --- Fase de Treino ---
    model.train()
    running_loss = 0.0
 
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
 
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
 
    # O scheduler usa a AUC calculada na validação logo abaixo
    scheduler.step(auc)
 
    # --- Fase de Validação ---
    model.eval()
    val_loss         = 0.0
    all_true_labels  = []
    all_probs        = []
    all_preds        = []
 
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
 
            outputs = model(images)
            v_loss  = criterion(outputs, labels)
            val_loss += v_loss.item()
 
            # Probabilidade da classe positiva (TEA) para calcular AUC
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
 
            all_probs.extend(probs.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
 
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss   = val_loss    / len(val_loader)
 
    # AUC pode falhar se apenas uma classe estiver presente no batch — tratamos com try/except
    try:
        auc = roc_auc_score(all_true_labels, all_probs) * 100
    except ValueError:
        auc = 0.0
 
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_auc'].append(auc)
 
    # --- Early Stopping e Checkpoint ---
    if auc > best_auc:
        best_auc           = auc
        epochs_sem_melhora = 0
        torch.save(model.state_dict(), caminho_modelo_final)
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
 
 
#   RESULTADOS FINAIS DO TREINO
# Distribuição das predições na última época de validação (útil para detectar
# colapso do modelo — ex.: prever sempre a mesma classe)
preds_tensor       = torch.tensor(all_preds)
unique_preds, counts = torch.unique(preds_tensor, return_counts=True)
chutes = dict(zip(unique_preds.tolist(), counts.tolist()))
 
print(f"\nUltima Epoca - Predicoes na Validacao: {chutes}")
print(
    f"Melhor AUC Alcançada: {best_auc:.1f}% "
    f"(Modelo salvo em {caminho_modelo_final})"
)
 
 

#   CURVAS DE APRENDIZADO 
plt.figure(figsize=(12, 4))
 
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Treino')
plt.plot(history['val_loss'],   label='Validacao')
plt.title('Curva de Loss')
plt.legend()
 
plt.subplot(1, 2, 2)
plt.plot(history['val_auc'], color='purple')
plt.title('Area Sob a Curva ROC (AUC)')
 
plt.tight_layout()
plt.show()
 
 
#   AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# Esta etapa só é executada UMA VEZ, ao final de todo o processo
# de desenvolvimento. O conjunto de Teste nunca foi visto durante o treino ou
# a seleção de hiperparâmetros, portanto a AUC obtida aqui é a estimativa
# honesta de generalização do modelo.
 
print("\n" + "=" * 50)
print("INICIANDO AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
print("=" * 50)
 
# Carrega o checkpoint do melhor modelo registrado durante o treino
modelo_teste = FCN_Apresentacao1D().to(DEVICE)
modelo_teste.load_state_dict(torch.load(caminho_modelo_final))
modelo_teste.eval()
 
test_loss        = 0.0
test_true_labels = []
test_probs       = []
test_preds       = []
 
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
 
        outputs = modelo_teste(images)
        loss    = criterion(outputs, labels)
        test_loss += loss.item()
 
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted = torch.max(outputs, 1)
 
        test_probs.extend(probs.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())
        test_preds.extend(predicted.cpu().numpy())
 
avg_test_loss = test_loss / len(test_loader)
 
try:
    test_auc = roc_auc_score(test_true_labels, test_probs) * 100
except ValueError:
    test_auc = 0.0
 
# Distribuição das predições no Teste (verifica se o modelo não colapsou)
preds_test_tensor            = torch.tensor(test_preds)
unique_test_preds, counts_test = torch.unique(preds_test_tensor, return_counts=True)
chutes_teste = dict(zip(unique_test_preds.tolist(), counts_test.tolist()))
 
print(f"Loss no Teste           : {avg_test_loss:.4f}")
print(f"Predicoes no Teste      : {chutes_teste}")
print(f">>> AUC FINAL REAL (TESTE): {test_auc:.1f}% <<<")
 
