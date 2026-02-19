import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast # Para treinar mais rápido

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


#CONFIGURANDO CAMINHOS E PARÂMETROS
BASE_DIR = '/content/drive/Shareddrives/Projeto_TEA/Dados_TEA_V2/ABIDE_pcp'
CSV_PATH = f'{BASE_DIR}/Phenotypic_V1_0b_preprocessed1.csv'
IMGS_DIR = f'{BASE_DIR}/cpac/filt_noglobal'

# Locais de Cache Local (Colab)
CACHE_TRAIN = '/content/fc_cache_train'
CACHE_VAL   = '/content/fc_cache_val'

# Locais de Backup no Drive (Opcional)
DRIVE_TRAIN = f'{BASE_DIR}/fc_cache_train'
DRIVE_VAL   = f'{BASE_DIR}/fc_cache_val'

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Pipeline FC no {DEVICE}")


#LIMPEZA DE CACHE ANTIGO
print("🧹 Limpando cache antigo para regenerar com Fisher-Z...")
if os.path.exists(CACHE_TRAIN): shutil.rmtree(CACHE_TRAIN)
if os.path.exists(CACHE_VAL): shutil.rmtree(CACHE_VAL)
print("Cache limpo.")


#PREPARANDO ATLAS E MASCARADOR
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize='zscore_sample',
    verbose=0
)

corr = ConnectivityMeasure(kind='correlation')

# Leitura do CSV
df = pd.read_csv(CSV_PATH)
df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip().str.replace('no_filename', '')

label_map = {
    row.FILE_ID: int(row.DX_GROUP == 1)
    for _, row in df.iterrows()
    if row.FILE_ID != 'nan'
}

# Listagem dos Arquivos
subjects = []
for fname in os.listdir(IMGS_DIR):
    if not fname.endswith(('.nii', '.nii.gz')): continue
    
    file_id = fname.replace('_func_preproc.nii.gz', '').replace('_func_preproc.nii', '')
    
    if file_id in label_map:
        subjects.append({
            "id": file_id,
            "path": os.path.join(IMGS_DIR, fname),
            "label": label_map[file_id]
        })

print(f"Total de sujeitos encontrados: {len(subjects)}")




#DEFININDO O TAMANHO DO TESTE

NUM_SUJEITOS_TESTE = 150 

if len(subjects) > NUM_SUJEITOS_TESTE:
    print(f"MODO PILOTO: Limitando para os primeiros {NUM_SUJEITOS_TESTE} sujeitos.")
    subjects_teste = subjects[:NUM_SUJEITOS_TESTE]
else:
    subjects_teste = subjects

# Split
train_subj, val_subj = train_test_split(
    subjects_teste,
    test_size=0.2,
    random_state=42,
    stratify=[s["label"] for s in subjects_teste]
)

print(f"Split: {len(train_subj)} Treino | {len(val_subj)} Validação")

# Checagem de segurança
labels_train = [s['label'] for s in train_subj]
print(f"   Balanceamento Treino -> 0 (TC): {labels_train.count(0)} | 1 (TEA): {labels_train.count(1)}")



#DATASET PERSONALIZADO PARA GERAR MATRIZES DE CORRELAÇÃO COM FISHER Z-TRANSFORM E INTERPOLAR PARA 64x64
class ABIDEFCDataset(Dataset):
    def __init__(self, subjects, cache_dir, drive_dir=None):
        self.subjects = subjects
        self.cache_dir = cache_dir
        self.drive_dir = drive_dir
        os.makedirs(cache_dir, exist_ok=True)
        if drive_dir: os.makedirs(drive_dir, exist_ok=True)
        self._process_missing()

    def _process_missing(self):
        for subj in tqdm(self.subjects, desc="Gerando Matrizes"):
            sid = subj["id"]
            cache_file = f"{self.cache_dir}/{sid}.pt"

            if os.path.exists(cache_file): continue

            try:
                #Extrai Sinais
                ts = masker.fit_transform(subj["path"])
                
                #Gera Matriz de Correlação
                mat = corr.fit_transform([ts])[0]
                np.fill_diagonal(mat, 0)
                
                #MELHORIA CRÍTICA: FISHER Z-TRANSFORM ---
                # Transforma a distribuição de [-1, 1] para Gaussiana
                mat = np.arctanh(np.clip(mat, -0.99, 0.99)) 
                # --------------------------------------------

                #Converte para Tensor e Redimensiona
                tensor = torch.from_numpy(mat).float().unsqueeze(0) 
                
                # Interpola para 64x64 (padrão para CNNs pequenas)
                tensor = F.interpolate(
                    tensor.unsqueeze(0), 
                    size=(64, 64), 
                    mode="bilinear"
                ).squeeze(0)

            except Exception as e:
                print(f"⚠️ Erro em {sid}: {e}")
                tensor = torch.zeros((1, 64, 64))

            torch.save((tensor, subj["label"]), cache_file)
            
            
            if self.drive_dir: self._sync(cache_file)

    def _sync(self, local_file):
        dst = os.path.join(self.drive_dir, os.path.basename(local_file))
        if not os.path.exists(dst):
            try: shutil.copyfile(local_file, dst)
            except: pass

    def __len__(self): return len(self.subjects)
    def __getitem__(self, idx):
        sid = self.subjects[idx]["id"]
        return torch.load(f"{self.cache_dir}/{sid}.pt")

# Criando Loaders
train_dataset = ABIDEFCDataset(train_subj, cache_dir=CACHE_TRAIN, drive_dir=DRIVE_TRAIN)
val_dataset = ABIDEFCDataset(val_subj, cache_dir=CACHE_VAL, drive_dir=DRIVE_VAL)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)




