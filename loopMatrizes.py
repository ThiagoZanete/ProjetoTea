

#PARTE DO TREINAMENTO DA REDE
# Uma rede neural convolucional muito simples, projetada para evitar overfitting em um conjunto de dados pequeno. Ela tem apenas 2 blocos convolucionais seguidos por um classificador denso leve. O uso de BatchNorm e Dropout ajuda a regularizar o modelo.
class MicroFCN(nn.Module):
    def __init__(self):
        super(MicroFCN, self).__init__()
        
        # Rede Rasa (Shallow) para evitar Overfitting em dados escassos
        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.3), # Dropout ajuda a regularizar
            
            # Bloco 2
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        # Classificador
        # Entrada 64x64 -> MaxPool(2) -> 32x32 -> MaxPool(2) -> 16x16
        # 16 canais * 16 * 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 32), # Camada densa pequena
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5), # Dropout forte na decisão
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


#TREINO PILOTO COM O MODELO SIMPLES PARA VER SE A PIPELINE FUNCIONA
model = MicroFCN().to(DEVICE)

# Otimizador ajustado para regularização forte
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

num_epochs = 60 # Menos épocas necessárias para ver se funciona
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_acc = 0.0

print("\nIniciando teste piloto com MicroFCN")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
    # Validação
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(images)
                v_loss = criterion(outputs, labels)
            
            val_loss += v_loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    acc = 100 * correct / total
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(acc)
    
    scheduler.step(avg_val_loss)
    
    print(f"Ep {epoch+1:02d}: T-Loss {avg_train_loss:.4f} | V-Loss {avg_val_loss:.4f} | Acc {acc:.1f}%", end="")
    
    if acc > best_acc:
        best_acc = acc
        print(f"(Salvando Melhor Modelo: {best_acc:.1f}%)")
        
        torch.save(model.state_dict(), caminho_modelo_final) 
        export_weights_to_hex(model, caminho_arquivo_hexa)

    else:
        print("")


#resultados do teste piloto
print(f"\n🏆 Melhor Acurácia no Teste Piloto: {best_acc:.1f}%")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Treino')
plt.plot(history['val_loss'], label='Validação')
plt.title('Curva de Loss (Busque Validação caindo)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], color='green')
plt.title('Acurácia de Validação')
plt.show()