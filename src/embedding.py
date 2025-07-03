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
    backup_interval_cells: int = 100000,
    resume_from_backup: bool = True,
    # --- NUOVI PARAMETRI PER IL PARALLELISMO ---
    start_offset_cells: int = 0, # Inizio dell'intervallo di cellule per questa macchina
    end_offset_cells: int = -1   # Fine dell'intervallo di cellule per questa macchina (-1 per elaborare fino alla fine del dataset)
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
        # Carica i dati in modalità backed per efficienza di memoria
        gexpr_feature = sc.read_h5ad(data_path, backed='r+')
        print(f"Successfully loaded data from {data_path} with shape {gexpr_feature.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading .h5ad file: {e}")
        return

    # Calcola il range effettivo di cellule da processare per QUESTA macchina
    total_cells_in_dataset = gexpr_feature.shape[0]
    
    # Assicurati che gli offset siano validi
    actual_start_idx = max(0, start_offset_cells)
    if end_offset_cells == -1:
        actual_end_idx = total_cells_in_dataset
    else:
        actual_end_idx = min(total_cells_in_dataset, end_offset_cells)

    if actual_start_idx >= actual_end_idx:
        print(f"Warning: Start index ({actual_start_idx}) is greater or equal to end index ({actual_end_idx}). No cells to process for this range.")
        return

    num_cells_to_process = actual_end_idx - actual_start_idx
    print(f"This process will handle cells from index {actual_start_idx} to {actual_end_idx - 1} (total: {num_cells_to_process} cells).")

    # Crea una save_path specifica per questa porzione, per evitare sovrascritture
    # useremo un nome che include il range di cellule per differenziare gli output
    specific_save_path = os.path.join(save_path, f"part_{actual_start_idx}_to_{actual_end_idx - 1}")
    os.makedirs(specific_save_path, exist_ok=True)
    print(f"Saving outputs for this partition in: {specific_save_path}")

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

    # Liste per accumulare tutti gli embedding e ID (per il file finale di QUESTA PARTIZIONE)
    all_gene_embeddings_partition = []
    all_cell_ids_partition = []
    
    # Indice interno per la ripresa del backup relativo a QUESTA PARTIZIONE
    current_partition_progress_idx = 0

    # --- LOGICA DI RIPRESA DA BACKUP (ADATTATA PER PARTIZIONI) ---
    if resume_from_backup:
        # Cerca backup specifici per questa partizione
        backup_prefix = f"{task_name}_backup_{actual_start_idx}_{actual_end_idx-1}_"
        backup_files = sorted([f for f in os.listdir(specific_save_path) if f.startswith(backup_prefix) and f.endswith(".h5ad")])
        
        if backup_files:
            last_backup_file = backup_files[-1]
            try:
                # Il nome del file di backup includerà l'offset di inizio della partizione
                # Es. "embedding_task_backup_0_99999_10000.h5ad" -> riparte da 10000 all'interno di questa partizione
                last_progress_idx_str = last_backup_file.split('_')[-1].replace('.h5ad', '')
                current_partition_progress_idx = int(last_progress_idx_str)
                print(f"Resuming from backup file: {last_backup_file}. Starting from internal cell index {current_partition_progress_idx} within this partition.")
                
                # Carica il contenuto del backup nell'array finale della PARTIZIONE
                backup_adata = sc.read_h5ad(os.path.join(specific_save_path, last_backup_file))
                all_gene_embeddings_partition.append(backup_adata.X)
                all_cell_ids_partition.extend(backup_adata.obs_names.tolist())

            except Exception as e:
                print(f"Warning: Could not resume from backup {last_backup_file} due to error: {e}. Starting this partition from scratch.")
                current_partition_progress_idx = 0
                all_gene_embeddings_partition = []
                all_cell_ids_partition = []
        else:
            print("No existing backup files found for this partition. Starting this partition from scratch.")
    else:
        print("Resume from backup is disabled. Starting this partition from scratch.")

    # Nomi dei file finali per questa partizione
    strname_npy_partition = os.path.join(
        specific_save_path,
        f"{task_name}_{ckpt_name}_{input_type}_{output_type}_embedding_{target_high_resolution}_part_{actual_start_idx}_to_{actual_end_idx-1}.npy"
    )
    strname_h5ad_partition = os.path.join(
        specific_save_path,
        f"{task_name}_{ckpt_name}_{input_type}_{output_type}_embedding_{target_high_resolution}_part_{actual_start_idx}_to_{actual_end_idx-1}.h5ad"
    )
    print(f'Final .npy embeddings for this partition will be saved at: {strname_npy_partition}')
    print(f'Final .h5ad with embeddings and IDs for this partition will be saved at: {strname_h5ad_partition}')

    try:
        resolution_value = float(target_high_resolution[1:])
    except (ValueError, IndexError):
        print(f"Warning: Could not parse numerical resolution from '{target_high_resolution}'. Using 0.0.")
        resolution_value = 0.0

    print("Starting inference for this partition with batch processing...")
    
    cell_ids_full_dataset = gexpr_feature.obs_names.tolist()

    current_batch_embeddings = []
    current_batch_ids = []

    # Il loop ora itera sul range specifico per questa macchina
    # L'argomento `initial` di tqdm deve essere calcolato sull'indice globale per la barra di progresso
    tqdm_initial_value = (actual_start_idx + current_partition_progress_idx) // batch_size
    tqdm_total_value = (actual_end_idx - actual_start_idx) // batch_size
    
    # Il range di iterazione è `actual_start_idx` fino a `actual_end_idx`,
    # ma deve ripartire da `current_partition_progress_idx` all'interno della partizione.
    # Quindi, il vero inizio nel loop è `actual_start_idx + current_partition_progress_idx`
    for i in tqdm(range(actual_start_idx + current_partition_progress_idx, actual_end_idx, batch_size), 
                  initial=tqdm_initial_value, 
                  total=tqdm_total_value, 
                  desc=f"Processing batches for part {actual_start_idx}-{actual_end_idx-1}"):
        
        end_idx_current_batch = min(i + batch_size, actual_end_idx) # La fine del batch non deve superare end_offset_cells

        # Estrai il batch di cellule dal dataset globale
        batch_adata = gexpr_feature[i:end_idx_current_batch, :]
        batch_cell_ids = cell_ids_full_dataset[i:end_idx_current_batch] 

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

        # --- LOGICA DI SALVATAGGIO DI BACKUP INCREMENTALE (ADATTATA) ---
        # Controlla se abbiamo raggiunto l'intervallo di salvataggio O se è l'ultimo batch della PARTIZIONE
        # La condizione per l'ultimo batch ora è `end_idx_current_batch == actual_end_idx`
        if (len(current_batch_ids) % backup_interval_cells < batch_size and len(current_batch_ids) > 0) or (end_idx_current_batch == actual_end_idx):
            if current_batch_embeddings: 
                
                # Concatena gli embedding e gli ID accumulati in questo intervallo temporaneo
                combined_interval_embeddings = np.concatenate(current_batch_embeddings, axis=0)
                all_gene_embeddings_partition.append(combined_interval_embeddings)
                all_cell_ids_partition.extend(current_batch_ids) # Sposta da temp a finale della partizione

                # Crea l'oggetto AnnData per il backup
                # L'indice del backup riflette l'avanzamento all'interno di QUESTA PARTIZIONE
                obs_df_backup = pd.DataFrame(index=all_cell_ids_partition)
                backup_adata = sc.AnnData(X=np.concatenate(all_gene_embeddings_partition, axis=0), obs=obs_df_backup)
                
                # Definisci il nome del file di backup con l'indice della cellula locale alla partizione
                # e anche gli indici globali della partizione per chiarezza
                backup_filename = os.path.join(specific_save_path, f"{backup_prefix}{len(all_cell_ids_partition)}.h5ad")
                backup_adata.write(backup_filename)
                print(f"Backup saved for {len(all_cell_ids_partition)} cells in this partition at: {backup_filename}")
                
                # Resetta le liste temporanee per il prossimo intervallo
                current_batch_embeddings = []
                current_batch_ids = []
                
    # --- SALVATAGGIO FINALE PER QUESTA PARTIZIONE ---
    if current_batch_embeddings: # Se ci sono dati rimanenti dopo l'ultimo backup
        combined_interval_embeddings = np.concatenate(current_batch_embeddings, axis=0)
        all_gene_embeddings_partition.append(combined_interval_embeddings)
        all_cell_ids_partition.extend(current_batch_ids)


    final_gene_embeddings_partition = np.concatenate(all_gene_embeddings_partition, axis=0)
    print(f"Generated total embeddings for this partition with shape: {final_gene_embeddings_partition.shape}")
    
    np.save(strname_npy_partition, final_gene_embeddings_partition) 

    obs_df_partition = pd.DataFrame(index=all_cell_ids_partition)
    embedding_adata_partition = sc.AnnData(X=final_gene_embeddings_partition, obs=obs_df_partition)
    
    embedding_adata_partition.write(strname_h5ad_partition)

    print("Inference complete for this partition.")
    print(f"Combined .npy embeddings for this partition saved at: {strname_npy_partition}")
    print(f"Final AnnData with embeddings and cell IDs for this partition saved at: {strname_h5ad_partition}")
