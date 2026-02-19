import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# CONFIGURANDO CAMINHOS E PARÂMETROS
BASE_DIR = '/content/drive/Shareddrives/Projeto_TEA/Dados_TEA_V2/ABIDE_pcp'
CSV_PATH = f'{BASE_DIR}/Phenotypic_V1_0b_preprocessed1.csv'
IMGS_DIR = f'{BASE_DIR}/cpac/filt_noglobal'
# Local de Cache Único (Resolve o problema de reprocessamento se o split mudar)
CACHE_DIR = '/content/fc_cache'
DRIVE_CACHE = f'{BASE_DIR}/fc_cache'
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pipeline FC no {DEVICE}")



# PREPARANDO ATLAS SCHAEFER (100 Regiões Funcionais)
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize='zscore_sample',
    verbose=0
)

corr = ConnectivityMeasure(kind='correlation', standardize='zscore_sample')
# LEITURA DO CSV
df = pd.read_csv(CSV_PATH)
print(f"Total de sujeitos no CSV original: {len(df)}")




# FILTRO DE HETEROGENEIDADE: Isolando apenas os pacientes da NYU
# (Nota: Verifique se o nome exato da coluna no seu CSV é 'SITE_ID')
df = df[df['SITE_ID'] == 'NYU'] 
print(f"Total de sujeitos da NYU após filtro: {len(df)}")


df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip().str.replace('no_filename', '')

label_map = {
    row.FILE_ID: int(row.DX_GROUP == 1)
    for _, row in df.iterrows()
    if row.FILE_ID != 'nan'
}

# LISTAGEM DOS ARQUIVOS
subjects = []
for fname in os.listdir(IMGS_DIR):
    if not fname.endswith(('.nii', '.nii.gz')): continue
    
    file_id = fname.replace('_func_preproc.nii.gz', '').replace('_func_preproc.nii', '')
    
    # Como o label_map agora só tem IDs da NYU, ele vai ignorar todos os outros hospitais!
    if file_id in label_map:
        subjects.append({
            "id": file_id,
            "path": os.path.join(IMGS_DIR, fname),
            "label": label_map[file_id]
        })

print(f"Total de imagens da NYU encontradas no Drive: {len(subjects)}")


# DEFININDO O TAMANHO DO TESTE
NUM_SUJEITOS_TESTE = 150 

if len(subjects) > NUM_SUJEITOS_TESTE:
    print(f"MODO PILOTO: Limitando para os primeiros {NUM_SUJEITOS_TESTE} sujeitos.")
    subjects_teste = subjects[:NUM_SUJEITOS_TESTE]
else:
    subjects_teste = subjects

# SPLIT DE TREINO E VALIDAÇÃO
train_subj, val_subj = train_test_split(
    subjects,
    test_size=0.2,
    random_state=42,
    stratify=[s["label"] for s in subjects]
)

print(f"Split NYU: {len(train_subj)} Treino | {len(val_subj)} Validação")




class ABIDEFCDataset(Dataset):
    def __init__(self, subjects, cache_dir, drive_dir=None):
        self.subjects = subjects
        self.cache_dir = cache_dir
        self.drive_dir = drive_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.drive_dir: 
            os.makedirs(self.drive_dir, exist_ok=True)
            
        self._process_missing()

    def _process_missing(self):
        for subj in tqdm(self.subjects, desc="Verificando/Gerando Matrizes"):
            sid = subj["id"]
            cache_file = os.path.join(self.cache_dir, f"{sid}.pt")
            
            # Define o caminho do arquivo no Drive
            drive_file = os.path.join(self.drive_dir, f"{sid}.pt") if self.drive_dir else None
            # Se já está no cache local do Colab (mais rápido). Pula para o próximo.
            if os.path.exists(cache_file): 
                continue
                
            # Se o Colab reiniciou, não está no cache local, mas já tá no drive.
            # Copia do Drive para o Colab para economizar tempo de processamento.
            if drive_file and os.path.exists(drive_file):
                try:
                    shutil.copyfile(drive_file, cache_file)
                    continue # Já restaurou do Drive, pode ir para o próximo sujeito!
                except Exception as e:
                    print(f"Erro ao copiar do Drive para o Colab o sujeito {sid}: {e}")

            # Se não existe nem no Colab nem no Drive. É um sujeito novo! Processa do zero.
            try:
                # Extrai Sinais
                ts = masker.fit_transform(subj["path"])
                # Gera Matriz de Correlação
                mat = corr.fit_transform([ts])[0]
                np.fill_diagonal(mat, 0)

                # FISHER Z-TRANSFORM
                mat = np.arctanh(np.clip(mat, -0.99, 0.99))#transforma em uma distribuição gaussiana, o que ajuda o modelo a aprender melhor. O clip é para evitar valores infinitos. 
                
                # Converte para Tensor
                tensor = torch.from_numpy(mat).float().unsqueeze(0) 
                # Salva localmente
                torch.save((tensor, subj["label"]), cache_file)
                # Sincroniza salvando uma cópia no Drive para o futuro
                if self.drive_dir: 
                    self._sync(cache_file)

            except Exception as e:
                print(f"Erro no processamento de {sid}: {e}")







    def _sync(self, local_file):
        dst = os.path.join(self.drive_dir, os.path.basename(local_file))
        if not os.path.exists(dst):
            try: shutil.copyfile(local_file, dst)
            except: pass

    def __len__(self): 
        return len(self.subjects)
        
    def __getitem__(self, idx):
        sid = self.subjects[idx]["id"]
        return torch.load(os.path.join(self.cache_dir, f"{sid}.pt"), weights_only=False)



# CRIANDO LOADERS (Apontando sempre para o mesmo diretório de cache unificado)
train_dataset = ABIDEFCDataset(train_subj, cache_dir=CACHE_DIR, drive_dir=DRIVE_CACHE)
val_dataset = ABIDEFCDataset(val_subj, cache_dir=CACHE_DIR, drive_dir=DRIVE_CACHE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)