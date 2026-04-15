"""
    PRÉ-PROCESSAMENTO DE CONECTIVIDADE FUNCIONAL (FC) PARA ABIDE
    - Extrair séries temporais de ROIs (atlas Schaefer 100) a partir de imagens
      fMRI pré-processadas do dataset ABIDE.
    - Calcular matrizes de correlação (conectividade funcional) por sujeito.
    - Adicionar features de conectividade em nível de rede (network-level FC).
    - Harmonizar dados entre sites com ComBat para remover viés de aquisição.
    - Selecionar as features mais discriminativas com SelectKBest (f_classif).
    - Normalizar os dados e exportar DataLoaders prontos para treino.

"""

import os
import time
import shutil
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

from collections import Counter
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from tqdm.auto import tqdm
from neuroCombat import neuroCombat
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


#   CONFIGURAÇÕES GLOBAIS E CAMINHOS
# Caminhos para os dados brutos e cache
BASE_DIR    = '/content/drive/Shareddrives/Projeto_TEA/Dados_TEA_V2/ABIDE_pcp'
CSV_PATH    = f'{BASE_DIR}/Phenotypic_V1_0b_preprocessed1.csv'
IMGS_DIR    = f'{BASE_DIR}/cpac/filt_noglobal'
CACHE_DIR   = '/content/cacheTeste'          # Cache local (RAM do Colab)
DRIVE_CACHE = f'{BASE_DIR}/cacheTeste'       # Cópia persistente no Google Drive

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pipeline FC no {DEVICE}")



#   REPRODUTIBILIDADE
def seed_everything(seed=42):
    """
    Congela todas as fontes de aleatoriedade do pipeline para garantir que
    os resultados sejam idênticos em execuções diferentes com os mesmos dados.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)




#   ATLAS E MEDIDA DE CONECTIVIDADE
def compute_network_fc(ts, atlas_labels):
    """
    Calcula a conectividade funcional em nível de rede (network-level FC).

    Motivação: além das conexões individuais entre pares de ROIs, é útil capturar
    o comportamento médio de redes inteiras (ex.: Default Mode, Visual, SomMot).
    Isso gera features complementares que refletem interações entre sistemas
    funcionais de larga escala.

    Estratégia:
        1. Agrupa as ROIs do atlas por rede (campo [2] do nome, ex: 'Vis', 'Default').
        2. Calcula o sinal médio temporal de cada rede.
        3. Computa a matriz de correlação entre as redes.
        4. Retorna apenas o triângulo superior (sem diagonal) como vetor de features.

    Parâmetros:
        ts           : np.array de shape (n_timepoints, n_rois) — séries temporais.
        atlas_labels : lista de strings com os nomes das ROIs do atlas Schaefer.

    Retorna:
        np.array 1D com os valores de correlação entre pares de redes.
    """
    # Agrupa os índices das ROIs por rede funcional
    networks = {}
    for i, label in enumerate(atlas_labels):
        name = label.decode() if isinstance(label, bytes) else str(label)
        # O campo [2] do nome Schaefer contém a rede (ex.: '7Networks_LH_Vis_1' → 'Vis')
        net = name.split('_')[2]
        if net not in networks:
            networks[net] = []
        networks[net].append(i)

    # Calcula o sinal médio temporal de cada rede
    network_ts = []
    for net in networks:
        idx = networks[net]
        network_ts.append(ts[:, idx].mean(axis=1))

    # Calcula a correlação entre os sinais médios das redes
    network_ts = np.array(network_ts).T
    net_corr = np.corrcoef(network_ts.T)

    # Extrai apenas o triângulo superior (pares únicos de redes)
    triu_idx = np.triu_indices(len(networks), k=1)
    return net_corr[triu_idx]


# Carrega o atlas Schaefer 2018 (100 ROIs, 7 redes de Yeo, resolução 2 mm)
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)

# Remove o rótulo 'Background' que o atlas inclui por padrão no índice 0.
# Sem essa limpeza, haveria um desalinhamento de índice entre ROIs e labels
# ao extrair as séries temporais (erro de index 100).
atlas_labels = []
for label in atlas.labels:
    name = label.decode() if isinstance(label, bytes) else str(label)
    if name != 'Background':
        atlas_labels.append(name)

# Medida de correlação de Pearson com z-score por amostra (padrão da literatura de FC)
corr = ConnectivityMeasure(kind='correlation', standardize="zscore_sample")




#   LEITURA DO CSV E MAPEAMENTO DE METADADOS
df = pd.read_csv(CSV_PATH)
print(f"Total de sujeitos no CSV original: {len(df)}")

# Normaliza a coluna FILE_ID: remove espaços e entradas inválidas ('no_filename')
df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip().str.replace('no_filename', '')

# Cria um dicionário {file_id → {label, site}} para consulta rápida durante o scan de imagens.
# DX_GROUP == 1 corresponde a diagnóstico de TEA (label = 1); controles ficam com label = 0.
info_map = {
    row.FILE_ID: {
        "label": int(row.DX_GROUP == 1),
        "site":  str(row.SITE_ID)
    }
    for _, row in df.iterrows()
    if row.FILE_ID != 'nan'
}






#   LISTAGEM E FILTRAGEM DE SUJEITOS
# Varre o diretório de imagens e monta a lista de sujeitos válidos (aqueles que tem entrada correspondente no CSV fenotípico). Cada sujeito é representado por um dicionário
subjects = []
for fname in os.listdir(IMGS_DIR):
    if not fname.endswith(('.nii', '.nii.gz')):
        continue

    file_id = fname.replace('_func_preproc.nii.gz', '').replace('_func_preproc.nii', '')

    if file_id in info_map:
        subjects.append({
            "id":    file_id,
            "path":  os.path.join(IMGS_DIR, fname),
            "label": info_map[file_id]["label"],
            "site":  info_map[file_id]["site"]
        })

print(f"Total de imagens encontradas no Drive: {len(subjects)}")

# Remove sites com poucos sujeitos: o ComBat exige um mínimo de amostras por batch para estimar os parâmetros de harmonização de forma confiável. 
MIN_PACIENTES_POR_SITE = 5
contagem_sites = Counter([s["site"] for s in subjects])

sites_removidos = [site for site, count in contagem_sites.items() if count < MIN_PACIENTES_POR_SITE]
if sites_removidos:
    print(f"Removendo sites com menos de {MIN_PACIENTES_POR_SITE} pacientes: {sites_removidos}")

subjects = [s for s in subjects if contagem_sites[s["site"]] >= MIN_PACIENTES_POR_SITE]
print(f"Total de imagens válidas após filtro: {len(subjects)}")






# SEÇÃO 6 — DIVISÃO TREINO / VALIDAÇÃO / TESTE
# Limita o número de sujeitos para execuções piloto mais rápidas
NUM_SUJEITOS_TESTE = 900
if len(subjects) > NUM_SUJEITOS_TESTE:
    print(f"MODO PILOTO: Limitando para os {NUM_SUJEITOS_TESTE} sujeitos aleatórios.")
    random.shuffle(subjects) # embaralha a lista para garantir uma amostra representativa dos sites e classes
    subjects_teste = subjects[:NUM_SUJEITOS_TESTE]
else:
    random.shuffle(subjects)
    subjects_teste = subjects

# Divisão estratificada por site E diagnóstico (site_label) para garantir
# que a distribuição de centros e classes seja proporcional em cada split.
# Proporção final: 60% Treino | 20% Validação | 20% Teste
try:
    # separa 20% para o conjunto de Teste Final (nunca visto durante treino)
    train_val_subj, test_subj = train_test_split(
        subjects_teste, test_size=0.2, random_state=42,
        stratify=[f"{s['site']}_{s['label']}" for s in subjects_teste]
    )
    # dos 80% restantes, separa 25% para Validação (= 20% do total)
    train_subj, val_subj = train_test_split(
        train_val_subj, test_size=0.25, random_state=42,
        stratify=[f"{s['site']}_{s['label']}" for s in train_val_subj]
    )
    print(f"Split: {len(train_subj)} Treino | {len(val_subj)} Val | {len(test_subj)} Teste")

except ValueError:
    # Fallback: estratifica apenas por diagnóstico quando algum site tem amostras demais escassas
    print("Aviso: Estratificando apenas por Diagnóstico devido à escassez de amostras em alguns sites.")
    train_val_subj, test_subj = train_test_split(
        subjects_teste, test_size=0.2, random_state=42,
        stratify=[s["label"] for s in subjects_teste]
    )
    train_subj, val_subj = train_test_split(
        train_val_subj, test_size=0.25, random_state=42,
        stratify=[s["label"] for s in train_val_subj]
    )



#   DATASET COM CACHE (EXTRAÇÃO E ARMAZENAMENTO DE FC)
class ABIDEFCDataset(Dataset):
    """
    Dataset PyTorch para o ABIDE com sistema de cache em dois níveis:
        - Nível 1 (rápido) : disco local do Colab (/content/cacheTeste)
        - Nível 2 (persistente): Google Drive (BASE_DIR/cacheTeste)

    Ao inicializar, verifica quais sujeitos ainda não foram processados e gera
    suas representações de FC. Sujeitos já processados são carregados do cache,
    evitando reprocessamento em caso de reinicialização do ambiente.

    Cada sujeito é representado por um vetor 1D que concatena:
        - Triângulo superior da matriz de correlação entre ROIs (Pearson + Fisher-Z)
        - Vetor de conectividade entre redes funcionais (network-level FC)
        - Ambos ordenados por rede para facilitar a aprendizagem de padrões locais
    """

    def __init__(self, subjects, cache_dir, drive_dir=None):
        self.subjects  = subjects
        self.cache_dir = cache_dir
        self.drive_dir = drive_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        if self.drive_dir:
            os.makedirs(self.drive_dir, exist_ok=True)

        self._process_missing()

    def _process_missing(self):
        """
        Itera sobre todos os sujeitos da lista e processa aqueles cujo arquivo
        de cache ainda não existe. A ordem de prioridade é:
            1. Cache local → já processado nesta sessão, pula direto.
            2. Cache no Drive → sessão reiniciada, copia para local e pula.
            3. Sem cache → processa do zero a partir da imagem fMRI no Drive.
        """
        for subj in tqdm(self.subjects, desc="Verificando/Gerando Matrizes"):
            sid        = subj["id"]
            cache_file = os.path.join(self.cache_dir, f"{sid}.pt")
            drive_file = os.path.join(self.drive_dir, f"{sid}.pt") if self.drive_dir else None

            # se o arquivo já está no cache local
            if os.path.exists(cache_file):
                continue

            # se oarquivo está no Drive copia para local
            if drive_file and os.path.exists(drive_file):
                try:
                    shutil.copyfile(drive_file, cache_file)
                    continue
                except Exception as e:
                    print(f"Erro ao copiar do Drive para o Colab o sujeito {sid}: {e}")

            # se o arquivo não está em nenhum cache, processa do zero
            try:
                time.sleep(0.5)

                # Copia a imagem para o disco local do Colab para leitura mais rápida
                caminho_local_temp = f"/content/temp_{sid}.nii.gz"
                shutil.copyfile(subj["path"], caminho_local_temp)

                # Lê o TR diretamente do cabeçalho NIfTI (índice 3 do zoom = dimensão temporal).
                # Alguns centros salvam o TR em milissegundos; a checagem abaixo converte para segundos.
                img    = nib.load(caminho_local_temp)
                tr_img = img.header.get_zooms()[3]
                if tr_img > 20:
                    tr_img = tr_img / 1000.0

                # Instancia o masker com o TR correto deste sujeito específico.
                # O TR varia entre centros do ABIDE, portanto não pode ser fixo globalmente.
                masker_dinamico = NiftiLabelsMasker(
                    labels_img=atlas.maps,
                    standardize='zscore_sample',
                    detrend=True,
                    low_pass=0.1,
                    high_pass=0.01,
                    t_r=float(tr_img)
                )

                # Extrai as séries temporais de cada ROI do atlas
                ts = masker_dinamico.fit_transform(caminho_local_temp)

                # Calcula e transforma a conectividade em nível de rede (Fisher-Z para normalizar)
                net_vec = compute_network_fc(ts, atlas_labels)
                net_vec = np.arctanh(np.clip(net_vec, -0.99, 0.99))

                # Remove o arquivo temporário para não lotar o disco do Colab
                os.remove(caminho_local_temp)

                # Calcula a matriz de correlação de Pearson entre todas as ROIs
                mat = corr.fit_transform([ts])[0]

                # Extrai apenas o triângulo superior da matriz (pares únicos de ROIs)
                triu_idx   = np.triu_indices(100, k=1)
                conn_vector = mat[triu_idx]

                # Atribui a cada ROI sua rede funcional (campo [2] do nome Schaefer)
                roi_network = []
                for label in atlas_labels:
                    name = label.decode() if isinstance(label, bytes) else str(label)
                    net  = name.split('_')[2]
                    roi_network.append(net)

                # Gera um rótulo de par de redes para cada aresta (ex.: ('Default', 'Vis'))
                edge_labels = []
                for i, j in zip(triu_idx[0], triu_idx[1]):
                    net1 = roi_network[i]
                    net2 = roi_network[j]
                    edge_labels.append(tuple(sorted([net1, net2])))

                # Reordena as arestas agrupando por par de redes.
                # Usar Python puro (sorted + lambda) evita o erro do numpy ao lidar com arrays 2D de tuplas, que não suportam comparação direta.
                edge_order  = sorted(range(len(edge_labels)), key=lambda x: edge_labels[x])
                conn_vector = conn_vector[edge_order]

                # Aplica Fisher-Z transform para linearizar a distribuição das correlações
                conn_vector = np.arctanh(np.clip(conn_vector, -0.99, 0.99))

                # Concatena as features de ROI-level e network-level em um único vetor
                conn_vector = np.concatenate([conn_vector, net_vec])

                # Salva o tensor localmente e sincroniza uma cópia no Drive
                tensor = torch.from_numpy(conn_vector).float()
                torch.save((tensor, subj["label"]), cache_file)
                if self.drive_dir:
                    self._sync(cache_file)

            except Exception as e:
                print(f"Erro no processamento de {sid}: {e}")
                if os.path.exists(f"/content/temp_{sid}.nii.gz"):
                    os.remove(f"/content/temp_{sid}.nii.gz")

    def _sync(self, local_file):
        # Copia o arquivo de cache local para o Drive se ainda não existir lá.
        dst = os.path.join(self.drive_dir, os.path.basename(local_file))
        if not os.path.exists(dst):
            try:
                shutil.copyfile(local_file, dst)
            except:
                pass

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        # Carrega o tensor de FC e o rótulo do sujeito a partir do cache local.
        sid = self.subjects[idx]["id"]
        return torch.load(os.path.join(self.cache_dir, f"{sid}.pt"), weights_only=False)



# SEÇÃO 8 — FUNÇÕES AUXILIARES
def load_all_from_cache(dataset, subject_list):
    """
    Itera sobre o dataset e coleta os vetores de FC (X), os rótulos (y) e os
    identificadores de site de cada sujeito em arrays NumPy separados.

    O site é necessário para a harmonização com ComBat (batch = site).
    """
    X, y, sites = [], [], []
    for i in range(len(dataset)):
        tensor, label = dataset[i]
        X.append(tensor.numpy())
        y.append(label)
        sites.append(subject_list[i]["site"])
    return np.array(X), np.array(y), np.array(sites)


def sparsify_fc(X, percent=0.2):
    """
    Aplica esparsificação na matriz de FC: zera todas as arestas cujo valor
    absoluto está abaixo do percentil (100 - percent*100) de cada sujeito.

    Isso elimina conexões fracas e potencialmente ruidosas, mantendo apenas
    as 'percent'% mais fortes de cada sujeito.
    """
    X_sparse = X.copy()
    for i in range(X.shape[0]):
        threshold = np.percentile(np.abs(X[i]), 100 * (1 - percent))
        mask = np.abs(X[i]) < threshold
        X_sparse[i][mask] = 0
    return X_sparse



# GERAÇÃO DOS DATASETS E EXTRAÇÃO DE FEATURES
print("\nVerificando Cache e Gerando Datasets...")
train_dataset = ABIDEFCDataset(train_subj, cache_dir=CACHE_DIR, drive_dir=DRIVE_CACHE)
val_dataset   = ABIDEFCDataset(val_subj,   cache_dir=CACHE_DIR, drive_dir=DRIVE_CACHE)
test_dataset  = ABIDEFCDataset(test_subj,  cache_dir=CACHE_DIR, drive_dir=DRIVE_CACHE)

print("\nCarregando dados da cache e extraindo sites...")
X_train, y_train, sites_train = load_all_from_cache(train_dataset, train_subj)
X_val,   y_val,   sites_val   = load_all_from_cache(val_dataset,   val_subj)
X_test,  y_test,  sites_test  = load_all_from_cache(test_dataset,  test_subj)

# O processamento fMRI pode gerar NaNs/Infs em sujeitos com séries temporais degeneradas
# (ex.: ROIs fora do campo de visão, covariância zero). Substituímos por 0 para não
# contaminar a harmonização e a seleção de features subsequentes.
print("Limpando possíveis NaNs e valores infinitos gerados pelo fMRI...")
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)




#   HARMONIZAÇÃO COM ComBat (REMOÇÃO DE EFEITO DE SITE)
#
# O ABIDE agrega dados de múltiplos centros com protocolos de aquisição distintos.
# O ComBat estima e remove o efeito de batch (site) preservando a variância
# biológica de interesse (diagnóstico de TEA).
#
# REGRA ANTIDATA-LEAKAGE:
#   - O ComBat é ajustado APENAS nos dados de treino.
#   - Validação e Teste recebem uma transformação "out-of-sample" que usa
#     apenas as estatísticas de site calculadas a partir do treino.


print("\nIniciando Harmonização com ComBat (Apenas Treino)...")
data_train_combat = X_train.T
covars_train = pd.DataFrame({
    'batch': sites_train,
    'label': y_train
})

combat_train = neuroCombat(
    dat=data_train_combat,
    covars=covars_train,
    batch_col='batch',
    categorical_cols=['label']
)
X_train_harmonized = combat_train['data'].T

# --- Harmonização Out-of-Sample para Validação ---
# Em vez de rodar o ComBat nos dados de validação (o que causaria leakage),
# padronizamos cada sujeito usando a média/std do SEU SITE no treino,
# e então reprojetamos para a distribuição global do treino harmonizado.
print("Aplicando correção de Site na Validação (Sem Data Leakage)...")
X_val_harmonized = X_val.copy()

# Estatísticas globais do treino pós-ComBat (alvo da reprojeção)
global_mean_train = X_train_harmonized.mean(axis=0)
global_std_train  = X_train_harmonized.std(axis=0) + 1e-8

# Estatísticas por site calculadas sobre o treino ORIGINAL (pré-ComBat)
site_stats = {}
for site in np.unique(sites_train):
    mask = (sites_train == site)
    site_stats[site] = {
        'mean': X_train[mask].mean(axis=0),
        'std':  X_train[mask].std(axis=0) + 1e-8
    }

# Aplica a transformação em cada sujeito de validação
for i, site in enumerate(sites_val):
    if site in site_stats:
        # Remove o viés do site e reprojeta para a distribuição global de treino
        z_val = (X_val[i] - site_stats[site]['mean']) / site_stats[site]['std']
        X_val_harmonized[i] = (z_val * global_std_train) + global_mean_train
    else:
        # Fallback para sites não vistos no treino: usa estatísticas globais do treino
        z_val = (X_val[i] - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
        X_val_harmonized[i] = (z_val * global_std_train) + global_mean_train

X_train = X_train_harmonized
X_val   = X_val_harmonized

# --- Harmonização Out-of-Sample para Teste (mesma lógica da validação) ---
X_test_harmonized = X_test.copy()
for i, site in enumerate(sites_test):
    if site in site_stats:
        z_test = (X_test[i] - site_stats[site]['mean']) / site_stats[site]['std']
        X_test_harmonized[i] = (z_test * global_std_train) + global_mean_train
    else:
        z_test = (X_test[i] - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
        X_test_harmonized[i] = (z_test * global_std_train) + global_mean_train

X_train = X_train_harmonized
X_val   = X_val_harmonized
X_test  = X_test_harmonized

# Salva os parâmetros de harmonização para uso futuro (ex.: inferência em novos dados)
harmonization_params = {
    'global_mean': global_mean_train,
    'global_std':  global_std_train,
    'site_stats':  site_stats
}
with open(os.path.join(BASE_DIR, 'combat_out_of_sample_params.pkl'), 'wb') as f:
    pickle.dump(harmonization_params, f)

print("Harmonização concluída! Seguindo para Seleção de Features...")

# O ComBat pode dividir features com variância zero (de ROIs constantes ou sujeitos ruins),
# gerando NaNs. O SelectKBest ignora essas features, mas é mais seguro zerá-las aqui.
print("Limpando NaNs residuais gerados pela divisão por zero no ComBat...")
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)


# Mantém as 512 conexões mais discriminativas segundo o f_classif.
# O fit é feito APENAS no treino; validação e teste recebem apenas o transform.
# Os índices selecionados são salvos para permitir interpretação posterior

print("Aplicando SelectKBest (512 conexões)...")
selector = SelectKBest(f_classif, k=512)

X_train = selector.fit_transform(X_train, y_train)
X_val   = selector.transform(X_val)
X_test  = selector.transform(X_test)

selected_indices = selector.get_support(indices=True)
np.save(os.path.join(BASE_DIR, "indices_selecionados.npy"), selected_indices)



# Padroniza cada feature para média 0 e desvio padrão 1.
# A média e o desvio são calculados APENAS no treino e aplicados
# igualmente na validação e no teste (sem leakage).

print("Aplicando Normalização Final...")
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std

# Salva os parâmetros de normalização para inferência futura em novos sujeitos
np.save(os.path.join(BASE_DIR, "norm_mean.npy"), mean)
np.save(os.path.join(BASE_DIR, "norm_std.npy"),  std)



#   CRIAÇÃO DOS DATALOADERS
tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.long)
tensor_X_val   = torch.tensor(X_val,   dtype=torch.float32)
tensor_y_val   = torch.tensor(y_val,   dtype=torch.long)
tensor_X_test  = torch.tensor(X_test,  dtype=torch.float32)
tensor_y_test  = torch.tensor(y_test,  dtype=torch.long)

# DataLoaders em memória (TensorDataset) são mais rápidos do que ler do disco a cada batch
train_loader = DataLoader(TensorDataset(tensor_X_train, tensor_y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(tensor_X_val,   tensor_y_val),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(tensor_X_test,  tensor_y_test),  batch_size=BATCH_SIZE)

print("Pré-processamento concluído! Dados prontos para a rede Conv1D.")