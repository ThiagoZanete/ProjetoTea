import os
import time
import shutil
import pandas as pd
import numpy as np
import torch
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from sklearn.feature_selection import SelectKBest, f_classif
from torch.utils.data import TensorDataset

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# caminhos 
BASE_DIR = '/content/drive/Shareddrives/Projeto_TEA/Dados_TEA_V2/ABIDE_pcp'
CSV_PATH = f'{BASE_DIR}/Phenotypic_V1_0b_preprocessed1.csv'
IMGS_DIR = f'{BASE_DIR}/cpac/filt_noglobal'
#cache
CACHE_DIR = '/content/fc_cache_1d'
DRIVE_CACHE = f'{BASE_DIR}/fc_cache_1d'
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pipeline FC no {DEVICE}")



# atlas
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)


corr = ConnectivityMeasure(kind='correlation', standardize= "zscore_sample")

#
# LEITURA DO CSV
df = pd.read_csv(CSV_PATH)
print(f"Total de sujeitos no CSV original: {len(df)}")


# df = df[df['SITE_ID'] == 'NYU'] 

df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip().str.replace('no_filename', '')


info_map = {
    row.FILE_ID: {
        "label": int(row.DX_GROUP == 1), 
        "site": str(row.SITE_ID)
    }
    for _, row in df.iterrows()
    if row.FILE_ID != 'nan'
}

# listagem
subjects = []
for fname in os.listdir(IMGS_DIR):
    if not fname.endswith(('.nii', '.nii.gz')): continue

    file_id = fname.replace('_func_preproc.nii.gz', '').replace('_func_preproc.nii', '')

    # ids válidos e presentes no info_map
    if file_id in info_map:
        subjects.append({
            "id": file_id,
            "path": os.path.join(IMGS_DIR, fname),
            "label": info_map[file_id]["label"],
            "site": info_map[file_id]["site"] 
        })

print(f"Total de imagens encontradas no Drive: {len(subjects)}")

# tamanho teste
NUM_SUJEITOS_TESTE = 400 

if len(subjects) > NUM_SUJEITOS_TESTE:
    print(f"MODO PILOTO: Limitando para os primeiros {NUM_SUJEITOS_TESTE} sujeitos.")
    subjects_teste = subjects[:NUM_SUJEITOS_TESTE]
else:
    subjects_teste = subjects

# SPLIT COM TRATAMENTO DE ERRO
# Como temos muitos centros, algum deles pode ter apenas 1 paciente de uma classe,
try:
    train_subj, val_subj = train_test_split(
        subjects_teste,
        test_size=0.2,
        random_state=42,
        stratify=[f"{s['site']}_{s['label']}" for s in subjects_teste]
    )
    print(f"Split Multi-Site: {len(train_subj)} Treino | {len(val_subj)} Validação (Estratificado por Site e Diagnóstico)")
except ValueError:
    print("Aviso: Algum centro médico tem amostras insuficientes para estratificação conjunta. Estratificando apenas por Diagnóstico.")
    train_subj, val_subj = train_test_split(
        subjects_teste,
        test_size=0.2,
        random_state=42,
        stratify=[s["label"] for s in subjects_teste]
    )
    print(f"Split Multi-Site: {len(train_subj)} Treino | {len(val_subj)} Validação (Estratificado apenas por Diagnóstico)")


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
            # Se já está no cache local do Colab. Pula para o próximo.
            if os.path.exists(cache_file):
                continue

            # Se o Colab reiniciou, não está no cache local, mas já tá no drive.
            # Copia do Drive para o Colab para economizar tempo de processamento.
            if drive_file and os.path.exists(drive_file):
                try:
                    shutil.copyfile(drive_file, cache_file)
                    continue
                except Exception as e:
                    print(f"Erro ao copiar do Drive para o Colab o sujeito {sid}: {e}")

            try:
                time.sleep(0.5)
                # Copia a imagem do Drive para a máquina local do Colab
                caminho_local_temp = f"/content/temp_{sid}.nii.gz"
                shutil.copyfile(subj["path"], caminho_local_temp)

                # Lendo o tr direto do cabeçalho da imagem ---
                img = nib.load(caminho_local_temp)
                # Em imagens fMRI (4D), o 4º elemento do zoom (índice 3) é o Tempo de Repetição
                tr_img = img.header.get_zooms()[3] 
                
                # Tratamento de segurança pq alguns centros salvam o TR em milissegundos 
                if tr_img > 20: 
                    tr_img = tr_img / 1000.0
                
                # Instancia o masker dinamicamente com o TR correto deste paciente específico
                masker_dinamico = NiftiLabelsMasker(
                    labels_img=atlas.maps,
                    standardize='zscore_sample',
                    detrend=True,
                    low_pass=0.1,
                    high_pass=0.01,
                    t_r=float(tr_img)
                )

                # Extrai Sinais usando o arquivo local e o masker dinâmico
                ts = masker_dinamico.fit_transform(caminho_local_temp)
            

                # Apaga a imagem temporária para não lotar o disco do Colab
                os.remove(caminho_local_temp)

                # Gera Matriz de Correlação
                mat = corr.fit_transform([ts])[0]

                # EXTRAÇÃO APENAS DO TRIÂNGULO SUPERIOR
                triu_idx = np.triu_indices(100, k=1)
                conn_vector = mat[triu_idx]

                # fisher z-transform para estabilizar a variância das correlações no vetor
                conn_vector = np.arctanh(np.clip(conn_vector, -0.99, 0.99))

                # Converte para Tensor
                tensor = torch.from_numpy(conn_vector).float()

                # Salva localmente
                torch.save((tensor, subj["label"]), cache_file)

                # Sincroniza salvando uma cópia no Drive para o futuro
                if self.drive_dir:
                    self._sync(cache_file)

            except Exception as e:
                print(f"Erro no processamento de {sid}: {e}")
                # Se der erro no meio, garante que o arquivo temporário será apagado
                if os.path.exists(f"/content/temp_{sid}.nii.gz"):
                    os.remove(f"/content/temp_{sid}.nii.gz")







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


# passa da cache para emoria
def load_all_from_cache(dataset):
    X, y = [], []
    for i in range(len(dataset)):
        tensor, label = dataset[i]
        X.append(tensor.numpy())
        y.append(label)
    return np.array(X), np.array(y)

print("\nCarregando dados para aplicar Seleção de Features...")
train_dataset = ABIDEFCDataset(train_subj, cache_dir=CACHE_DIR, drive_dir=DRIVE_CACHE)
val_dataset = ABIDEFCDataset(val_subj, cache_dir=CACHE_DIR, drive_dir=DRIVE_CACHE)

X_train, y_train = load_all_from_cache(train_dataset)
X_val, y_val = load_all_from_cache(val_dataset)

# Select Kbest
print("Aplicando SelectKBest (reduzindo para as 256 melhores conexões)...")
selector = SelectKBest(f_classif, k=256)
X_train = selector.fit_transform(X_train, y_train)
X_val = selector.transform(X_val)

# Salvar índices para o futuro 
selected_indices = selector.get_support(indices=True)
np.save(os.path.join(BASE_DIR, "indices_selecionados.npy"), selected_indices)

#normalização
print("Aplicando Normalização...")
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-6

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

np.save(os.path.join(BASE_DIR, "norm_mean.npy"), mean)
np.save(os.path.join(BASE_DIR, "norm_std.npy"), std)

#tensores e dataloader
tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.long)
tensor_X_val = torch.tensor(X_val, dtype=torch.float32)
tensor_y_val = torch.tensor(y_val, dtype=torch.long)

# Dataset na memória é mais rápido para treinar
train_loader = DataLoader(TensorDataset(tensor_X_train, tensor_y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(tensor_X_val, tensor_y_val), batch_size=BATCH_SIZE)

print("Pré-processamento concluído! Dados prontos para a rede Conv1D.")