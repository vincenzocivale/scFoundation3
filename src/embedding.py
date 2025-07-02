import random
import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
import h5py
from src.load import load_model_frommmf, gatherData

def process_gene_expression(
    data_path: str,
    ckpt_path: str = './models/models.ckpt',
    save_path: str = './output',
    task_name: str = 'embedding_task',
    ckpt_name: str = 'model',
    input_type: str = 'gene_expression',
    output_type: str = 'embedding',
    target_high_resolution: str = 'R1',
    pool_type: str = 'all',
    batch_size: int = 1000,
    seed: int = 0,
    use_fp16: bool = False # Aggiunto il parametro use_fp16
):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        gexpr_feature = sc.read_h5ad(data_path, backed='r+')
        print(f"Successfully loaded data from {data_path} with shape {gexpr_feature.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading .h5ad file: {e}")
        return

    os.makedirs(save_path, exist_ok=True)

    key = 'cell'

    try:
        pretrainmodel, pretrainconfig = load_model_frommmf(ckpt_path, key)
        pretrainmodel.eval()
        pretrainmodel.to(device)
        if use_fp16 and device.type == 'cuda':
            pretrainmodel.half()
            print("Model converted to half precision (float16).")
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return

    all_gene_embeddings = []

    strname = os.path.join(
        save_path,
        f"{task_name}_{ckpt_name}_{input_type}_{output_type}_embedding_{target_high_resolution}_resolution.npy"
    )
    print(f'Saving embeddings at: {strname}')

    try:
        resolution_value = float(target_high_resolution[1:])
    except (ValueError, IndexError):
        print(f"Warning: Could not parse numerical resolution from '{target_high_resolution}'. Using 0.0.")
        resolution_value = 0.0

    print("Starting inference with batch processing...")
    num_cells = gexpr_feature.shape[0]

    for i in tqdm(range(0, num_cells, batch_size), desc="Processing batches"):
        end_idx = min(i + batch_size, num_cells)

        # Estrai il batch di cellule (l'oggetto AnnData in modalità backed carica i dati in questo punto)
        batch_adata = gexpr_feature[i:end_idx, :] # Ora batch_adata è un AnnData con i dati del batch

        # Prepara i dati del batch
        batch_gene_x_list = []

        # Accedi alla matrice X del batch
        if issparse(batch_adata.X):
            # Se la matrice X è sparsa, sommiamo lungo l'asse dei geni per ogni cellula
            # Non è necessario convertire in toarray() l'intero batch, solo per la riga attuale
            batch_data_rows = batch_adata.X # Questo è ancora un oggetto CSR/CSC sparso
        else:
            batch_data_rows = batch_adata.X # Questo è un array NumPy denso

        for row_idx_in_batch in range(batch_adata.shape[0]):
            # Estrai la riga corrente come array denso per la somma e il tolist
            current_cell_data = batch_data_rows[row_idx_in_batch, :]

            # Calcola il totalcount sommando tutti i valori della riga
            # Gestisce sia sparse che dense
            totalcount = current_cell_data.sum()

            # Gestione di totalcount = 0 per evitare log10(0)
            log_total_count = np.log10(totalcount) if totalcount > 0 else np.log10(1)

            # Assicurati che tmpdata sia un elenco di numeri float
            if issparse(current_cell_data):
                tmpdata = current_cell_data.toarray().flatten().tolist()
            else:
                tmpdata = current_cell_data.tolist()

            batch_gene_x_list.append(tmpdata + [resolution_value, log_total_count])

        # Converti la lista di liste in un tensore unico per il batch
        if use_fp16 and device.type == 'cuda':
            pretrain_gene_x_batch = torch.tensor(batch_gene_x_list).to(device).half()
        else:
            pretrain_gene_x_batch = torch.tensor(batch_gene_x_list).to(device).float()

        with torch.no_grad():
            data_gene_ids = torch.arange(19266, device=pretrain_gene_x_batch.device).repeat(pretrain_gene_x_batch.shape[0], 1)

            value_labels = pretrain_gene_x_batch > 0
            x, x_padding = gatherData(pretrain_gene_x_batch, value_labels, pretrainconfig['pad_token_id'])

            position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])

            # *** CORREZIONE PER L'ERRORE FLOAT/HALF ***
            # Assicurati che l'input a token_emb sia del tipo corretto (half se il modello è half)
            if use_fp16 and device.type == 'cuda':
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).half(), output_weight = 0)
            else:
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)

            position_emb = pretrainmodel.pos_emb(position_gene_ids)
            # Anche position_emb deve essere dello stesso tipo (half() se il modello è half())
            if use_fp16 and device.type == 'cuda':
                position_emb = position_emb.half()

            x += position_emb
            geneemb = pretrainmodel.encoder(x, x_padding)

            geneemb1 = geneemb[:,-1,:]
            geneemb2 = geneemb[:,-2,:]
            geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
            geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)

            if pool_type == 'all':
                geneembmerge = torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
            elif pool_type == 'max':
                geneembmerge, _ = torch.max(geneemb, dim=1)
            else:
                raise ValueError("pool_type must be 'all' or 'max'")

            all_gene_embeddings.append(geneembmerge.detach().cpu().numpy())

        # Pulizia della memoria dopo ogni batch
        # Elimina le variabili locali del ciclo per liberare memoria
        del batch_adata # Rimuovi anche questa
        del batch_gene_x_list
        del pretrain_gene_x_batch
        del data_gene_ids
        del value_labels
        del x, x_padding, position_gene_ids, position_emb, geneemb, geneemb1, geneemb2, geneemb3, geneemb4, geneembmerge
        torch.cuda.empty_cache()

    final_gene_embeddings = np.concatenate(all_gene_embeddings, axis=0)
    print(f"Generated total embeddings with shape: {final_gene_embeddings.shape}")
    np.save(strname, final_gene_embeddings)
    print("Inference complete and embeddings saved.")
