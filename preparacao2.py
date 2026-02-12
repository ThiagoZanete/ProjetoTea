import os
import time
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






BASE_DIR = '/content/drive/Shareddrives/Projeto_TEA/Dados_TEA_V2/ABIDE_pcp'
CSV_PATH = f'{BASE_DIR}/Phenotypic_V1_0b_preprocessed1.csv'
IMGS_DIR = f'{BASE_DIR}/cpac/filt_noglobal'

CACHE_TRAIN = '/content/fc_cache_train'
CACHE_VAL   = '/content/fc_cache_val'

DRIVE_TRAIN = f'{BASE_DIR}/fc_cache_train'
DRIVE_VAL   = f'{BASE_DIR}/fc_cache_val'

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️ Pipeline FC no {DEVICE}")





#atlas + masker
atlas = datasets.fetch_atlas_harvard_oxford(#obtém o atlas de Harvard-Oxford, que é um conjunto de regiões cerebrais pré-definidas usadas para análise de conectividade funcional.
    'cort-maxprob-thr25-2mm'
)

masker = NiftiLabelsMasker(#O NiftiLabelsMasker é uma ferramenta do Nilearn que extrai sinais médios de regiões específicas do cérebro, definidas por um atlas. Ele transforma os dados 4D (espaço + tempo) em uma matriz 2D (regiões x tempo).
    labels_img=atlas.maps,
    standardize='zscore_sample',
    verbose=0
)

corr = ConnectivityMeasure(kind='correlation')#O ConnectivityMeasure calcula medidas de conectividade funcional, como correlação, entre as regiões cerebrais. Ele transforma os sinais extraídos pelo masker em uma matriz de conectividade (regiões x regiões), onde cada elemento representa a força da conexão entre duas regiões.






#mapeamento

df = pd.read_csv(CSV_PATH)#Lê o arquivo CSV contendo os dados fenotípicos dos sujeitos

df['FILE_ID'] = (#Limpa a coluna FILE_ID, removendo espaços e substituindo 'no_filename' por vazio
    df['FILE_ID']
    .astype(str)
    .str.strip()
    .str.replace('no_filename', '')
)

label_map = {#Cria um dicionário que mapeia cada FILE_ID para um rótulo binário (1 para DX_GROUP == 1, 0 caso contrário)
    row.FILE_ID: int(row.DX_GROUP == 1)
    for _, row in df.iterrows()
    if row.FILE_ID != 'nan'
}


subjects = []

for fname in os.listdir(IMGS_DIR):#Percorre os arquivos na pasta de imagens pré-processadas, filtrando apenas os arquivos NIfTI e associando cada arquivo a um rótulo usando o label_map criado anteriormente.
    if not fname.endswith(('.nii', '.nii.gz')):
        continue

    file_id = fname.replace('_func_preproc.nii.gz', '').replace('_func_preproc.nii', '')

    if file_id in label_map:#Se o FILE_ID extraído do nome do arquivo existir no label_map, adiciona um dicionário com o ID, caminho e rótulo do sujeito à lista de sujeitos válidos.
        subjects.append({
            "id": file_id,
            "path": os.path.join(IMGS_DIR, fname),
            "label": label_map[file_id]
        })

print(f"📂 Sujeitos válidos: {len(subjects)}")








#split dos dados em treino e validação, garantindo que a proporção de rótulos seja mantida em ambos os conjuntos.
train_subj, val_subj = train_test_split(
    subjects,
    test_size=0.2,
    random_state=42,
    stratify=[s["label"] for s in subjects]
)





#criação e salvamento do dataset no drive 
class ABIDEFCDataset(Dataset):
    def __init__(self, subjects, cache_dir, drive_dir=None):#O construtor do dataset recebe a lista de sujeitos, o diretório de cache para salvar os arquivos processados e um diretório opcional no Google Drive para sincronização. Ele cria os diretórios necessários e chama o método _process_missing para processar os sujeitos que ainda não foram processados.
        self.subjects = subjects
        self.cache_dir = cache_dir
        self.drive_dir = drive_dir

        os.makedirs(cache_dir, exist_ok=True)#Cria o diretório de cache local, se não existir
        if drive_dir:
            os.makedirs(drive_dir, exist_ok=True)

        self._process_missing()

    def _process_missing(self):#Este método percorre os sujeitos e verifica se o arquivo processado já existe no cache. Se não existir, ele processa o sujeito usando o masker para extrair os sinais médios das regiões cerebrais e o corr para calcular a matriz de conectividade. A matriz é então redimensionada para 64x64 usando interpolação bilinear e salva como um tensor PyTorch. Se ocorrer algum erro durante o processamento, um tensor de zeros é salvo em vez disso. Após salvar o arquivo localmente, ele é sincronizado com o Google Drive, se um diretório de drive for fornecido.
        for subj in tqdm(self.subjects, desc="Processando FC"):
            sid = subj["id"]
            cache_file = f"{self.cache_dir}/{sid}.pt"

            if os.path.exists(cache_file):
                continue

            try:
                ts = masker.fit_transform(subj["path"])#Usa atlas para extrair a atividade média de cada região ao longo do temp, dessa forma trransforma o vídeo 4D em uma tabela de sinais.
                mat = corr.fit_transform([ts])[0]#Calcula a matriz de correlação. Se a região A sobe atividade quando a B sobe, a conexão é forte.
                np.fill_diagonal(mat, 0)

                tensor = torch.from_numpy(mat).float().unsqueeze(0)#Converte a matriz de conectividade em um tensor PyTorch 
                tensor = F.interpolate(#só para redimensionar a matriz para 64x64
                    tensor.unsqueeze(0),
                    size=(64, 64),
                    mode="bilinear"
                ).squeeze(0)

            except Exception as e:
                print(f"⚠️ {sid}: {e}")
                tensor = torch.zeros((1, 64, 64))

            torch.save((tensor, subj["label"]), cache_file)

            if self.drive_dir:
                self._sync(cache_file)

    def _sync(self, local_file):#Este método sincroniza um arquivo local com o Google Drive. Ele constrói o caminho de destino no drive e copia o arquivo local para lá, se ele ainda não existir. Se ocorrer algum erro durante a cópia, ele é ignorado.
        dst = os.path.join(self.drive_dir, os.path.basename(local_file))
        if os.path.exists(dst):
            return
        try:
            shutil.copyfile(local_file, dst)
        except Exception:
            pass

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid = self.subjects[idx]["id"]
        return torch.load(f"{self.cache_dir}/{sid}.pt")









train_dataset = ABIDEFCDataset(#Cria o dataset de treino usando a classe ABIDEFCDataset, passando os sujeitos de treino, o diretório de cache para treino e o diretório de drive para treino.
    train_subj,
    cache_dir=CACHE_TRAIN,
    drive_dir=DRIVE_TRAIN
)

val_dataset = ABIDEFCDataset(#Cria o dataset de validação usando a classe ABIDEFCDataset, passando os sujeitos de validação, o diretório de cache para validação e o diretório de drive para validação.
    val_subj,
    cache_dir=CACHE_VAL,
    drive_dir=DRIVE_VAL
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
