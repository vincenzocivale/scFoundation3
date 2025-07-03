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
    use_fp16: bool = True,
    # --- NUOVI PARAMETRI PER IL SALVATAGGIO DI BACKUP ---
    backup_interval_cells: int = 100000, # Salva un backup ogni N cellule
    resume_from_backup: bool = True # Cerca i backup esistenti e riprendi
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

    # Liste per accumulare tutti gli embedding e ID (per il file finale)
    all_gene_embeddings_final = []
    all_cell_ids_final = []

    # Per tenere traccia dell'indice da cui ripartire
    start_cell_idx = 0

    # --- LOGICA DI RIPRESA DA BACKUP ---
    if resume_from_backup:
        backup_files = sorted([f for f in os.listdir(save_path) if f.startswith(f"{task_name}_backup_") and f.endswith(".h5ad")])
        if backup_files:
            last_backup_file = backup_files[-1]
            try:
                # Il nome del file di backup dovrebbe contenere l'indice della cellula finale del backup
                # Es. "embedding_task_backup_0_100000.h5ad" -> riparte da 100000
                last_end_idx_str = last_backup_file.split('_')[-1].replace('.h5ad', '')
                start_cell_idx = int(last_end_idx_str)
                print(f"Resuming from backup file: {last_backup_file}. Starting from cell index {start_cell_idx}.")

                # Carica il contenuto del backup nell'array finale
                backup_adata = sc.read_h5ad(os.path.join(save_path, last_backup_file))
                all_gene_embeddings_final.append(backup_adata.X)
                all_cell_ids_final.extend(backup_adata.obs_names.tolist())

            except Exception as e:
                print(f"Warning: Could not resume from backup {last_backup_file} due to error: {e}. Starting from scratch.")
                start_cell_idx = 0
                all_gene_embeddings_final = []
                all_cell_ids_final = []
        else:
            print("No existing backup files found. Starting from scratch.")
    else:
        print("Resume from backup is disabled. Starting from scratch.")

    strname_npy = os.path.join(
        save_path,
        f"{task_name}_{ckpt_name}_{input_type}_{output_type}_embedding_{target_high_resolution}_resolution.npy"
    )
    strname_h5ad = os.path.join(
        save_path,
        f"{task_name}_{ckpt_name}_{input_type}_{output_type}_embedding_{target_high_resolution}_resolution.h5ad"
    )
    print(f'Final .npy embeddings will be saved at: {strname_npy}')
    print(f'Final .h5ad with embeddings and IDs will be saved at: {strname_h5ad}')


    try:
        resolution_value = float(target_high_resolution[1:])
    except (ValueError, IndexError):
        print(f"Warning: Could not parse numerical resolution from '{target_high_resolution}'. Using 0.0.")
        resolution_value = 0.0

    print("Starting inference with batch processing...")
    num_cells = gexpr_feature.shape[0]
    cell_ids_full_dataset = gexpr_feature.obs_names.tolist()

    # Liste temporanee per gli embedding e ID del batch corrente (per il backup incrementale)
    current_batch_embeddings = []
    current_batch_ids = []

    # Loop principale che inizia da start_cell_idx
    for i in tqdm(range(start_cell_idx, num_cells, batch_size), initial=start_cell_idx // batch_size, total=num_cells // batch_size, desc="Processing batches"):
        end_idx = min(i + batch_size, num_cells)

        batch_adata = gexpr_feature[i:end_idx, :]
        batch_cell_ids = cell_ids_full_dataset[i:end_idx]

        batch_gene_x_list = []
        batch_data_rows = batch_adata.X

        for row_idx_in_batch in range(batch_adata.shape[0]):
            current_cell_data = batch_data_rows[row_idx_in_batch, :]
            totalcount = current_cell_data.sum()
            log_total_count = np.log10(totalcount) if totalcount > 0 else np.log10(1)

            if issparse(current_cell_data):
                tmpdata = current_cell_data.toarray().flatten().tolist()
            else:
                tmpdata = current_cell_data.tolist()

            batch_gene_x_list.append(tmpdata + [resolution_value, log_total_count])

        if use_fp16 and device.type == 'cuda':
            pretrain_gene_x_batch = torch.tensor(batch_gene_x_list).to(device).half()
        else:
            pretrain_gene_x_batch = torch.tensor(batch_gene_x_list).to(device).float()

        with torch.no_grad():
            data_gene_ids = torch.arange(19266, device=pretrain_gene_x_batch.device).repeat(pretrain_gene_x_batch.shape[0], 1)
            value_labels = pretrain_gene_x_batch > 0
            x, x_padding = gatherData(pretrain_gene_x_batch, value_labels, pretrainconfig['pad_token_id'])
            position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])

            if use_fp16 and device.type == 'cuda':
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).half(), output_weight = 0)
            else:
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)

            position_emb = pretrainmodel.pos_emb(position_gene_ids)
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

            # Aggiungi gli embedding e gli ID del batch alle liste temporanee
            current_batch_embeddings.append(geneembmerge.detach().cpu().numpy())
            current_batch_ids.extend(batch_cell_ids)

        # Pulizia della memoria dopo ogni batch
        del batch_adata
        del batch_gene_x_list
        del pretrain_gene_x_batch
        del data_gene_ids
        del value_labels
        del x, x_padding, position_gene_ids, position_emb, geneemb, geneemb1, geneemb2, geneemb3, geneemb4, geneembmerge
        torch.cuda.empty_cache()

        # --- LOGICA DI SALVATAGGIO DI BACKUP INCREMENTALE ---
        # Controlla se abbiamo raggiunto l'intervallo di salvataggio O se è l'ultimo batch
        if (len(all_cell_ids_final) + len(current_batch_ids)) % backup_interval_cells < batch_size or (end_idx == num_cells):
            if current_batch_embeddings: # Assicurati che ci siano dati da salvare

                # Concatena gli embedding e gli ID accumulati in questo intervallo
                combined_interval_embeddings = np.concatenate(current_batch_embeddings, axis=0)
                all_gene_embeddings_final.append(combined_interval_embeddings)
                all_cell_ids_final.extend(current_batch_ids)

                # Crea l'oggetto AnnData per il backup
                obs_df_backup = pd.DataFrame(index=all_cell_ids_final)
                backup_adata = sc.AnnData(X=np.concatenate(all_gene_embeddings_final, axis=0), obs=obs_df_backup)

                # Definisci il nome del file di backup con l'indice della cellula fino a cui siamo arrivati
                backup_filename = os.path.join(save_path, f"{task_name}_backup_{len(all_cell_ids_final)}.h5ad")
                backup_adata.write(backup_filename)
                print(f"Backup saved for {len(all_cell_ids_final)} cells at: {backup_filename}")

                # Resetta le liste temporanee per il prossimo intervallo
                current_batch_embeddings = []
                current_batch_ids = []

    # --- SALVATAGGIO FINALE ---
    # Se ci sono dati rimanenti dopo l'ultimo backup (o se non ci sono mai stati backup)
    if current_batch_embeddings:
        combined_interval_embeddings = np.concatenate(current_batch_embeddings, axis=0)
        all_gene_embeddings_final.append(combined_interval_embeddings)
        all_cell_ids_final.extend(current_batch_ids)


    final_gene_embeddings = np.concatenate(all_gene_embeddings_final, axis=0)
    print(f"Generated total embeddings with shape: {final_gene_embeddings.shape}")

    np.save(strname_npy, final_gene_embeddings)

    obs_df = pd.DataFrame(index=all_cell_ids_final)
    embedding_adata = sc.AnnData(X=final_gene_embeddings, obs=obs_df)

    embedding_adata.write(strname_h5ad)

    print("Inference complete.")
    print(f"Combined .npy embeddings saved at: {strname_npy}")
    print(f"Final AnnData with embeddings and cell IDs saved at: {strname_h5ad}")
