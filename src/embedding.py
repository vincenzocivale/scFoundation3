import random
import os
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm # Usa tqdm.notebook per una barra di progresso migliore nei notebook
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
    batch_size: int = 1000,  # Nuovo parametro per la dimensione del batch
    seed: int = 0
):
    """
    Processes gene expression data to generate embeddings using a pre-trained model,
    processing data in batches to manage memory efficiently.

    Args:
        data_path (str): Path to the input .h5ad gene expression data file.
        ckpt_path (str): Path to the pre-trained model checkpoint.
        save_path (str): Directory to save the output embeddings.
        task_name (str): Name of the current task, used for naming output files.
        ckpt_name (str): Name of the checkpoint, used for naming output files.
        input_type (str): Description of the input data type.
        output_type (str): Description of the output data type.
        target_high_resolution (str): Target high resolution, e.g., 'R1', 'R2'.
                                      The numeric part will be extracted.
        pool_type (str): Pooling type for generating the final embedding.
                         Can be 'all' (concatenates multiple pooled embeddings)
                         or 'max' (takes the max-pooled embedding).
        batch_size (int): Number of cells to process at a time.
        seed (int): Random seed for reproducibility.
    """
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

    # Load data metadata (useful for total count if not in X) or prepare for chunked reading if possible
    # For .h5ad, scanpy.read_h5ad loads everything. If data is too big for RAM,
    # consider Dask or AnnData's backed mode if your 'load' functions support it.
    # Here, we assume gexpr_feature is loaded once, but processing is chunked.
    try:
        gexpr_feature = sc.read_h5ad(data_path)
        print(f"Successfully loaded data from {data_path} with shape {gexpr_feature.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading .h5ad file: {e}")
        return

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    key = 'cell' # This 'key' seems fixed
    
    try:
        pretrainmodel, pretrainconfig = load_model_frommmf(ckpt_path, key)
        pretrainmodel.eval()
        pretrainmodel.to(device) # Move model to the selected device
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return

    all_gene_embeddings = [] # Lista per raccogliere gli embedding di tutti i batch
    
    # Construct output file name
    strname = os.path.join(
        save_path,
        f"{task_name}_{ckpt_name}_{input_type}_{output_type}_embedding_{target_high_resolution}_resolution.npy"
    )
    print(f'Saving embeddings at: {strname}')
    
    # Extract numerical part from target_high_resolution (e.g., 'R1' -> 1.0)
    try:
        resolution_value = float(target_high_resolution[1:])
    except (ValueError, IndexError):
        print(f"Warning: Could not parse numerical resolution from '{target_high_resolution}'. Using 0.0.")
        resolution_value = 0.0

    # Inference loop with batch processing
    print("Starting inference with batch processing...")
    num_cells = gexpr_feature.shape[0]
    
    for i in tqdm(range(0, num_cells, batch_size), desc="Processing batches"):
        end_idx = min(i + batch_size, num_cells)
        
        # Estrai il batch di cellule
        batch_data_subset = gexpr_feature[i:end_idx, :]

        # Prepara i dati del batch
        batch_gene_x_list = []
        batch_total_counts = []

        if issparse(batch_data_subset.X):
            # Processa le colonne singolarmente o converti l'intero batch in un array denso
            # Attenzione: .toarray() su un batch grande potrebbe ancora essere problematico
            # Considera di processare riga per riga se un singolo batch è ancora troppo grande dopo .toarray()
            batch_data_array = batch_data_subset.X.toarray()
            
            for row_idx in range(batch_data_array.shape[0]):
                tmpdata = batch_data_array[row_idx, :].flatten().tolist()
                totalcount = batch_data_array[row_idx, -1] # Assumendo l'ultima colonna è il totalcount
                batch_gene_x_list.append(tmpdata + [resolution_value, np.log10(totalcount)])
        else:
            # Se i dati sono densi (pandas DataFrame o numpy array)
            for _, row in batch_data_subset.to_df().iterrows(): # Converti in DataFrame per iterare facilmente
                tmpdata = row.tolist()
                totalcount = row.iloc[-1] # Assumendo l'ultima colonna è il totalcount
                batch_gene_x_list.append(tmpdata + [resolution_value, np.log10(totalcount)])

        # Converti la lista di liste in un tensore unico per il batch
        pretrain_gene_x_batch = torch.tensor(batch_gene_x_list).to(device).float()

        with torch.no_grad():
            data_gene_ids = torch.arange(19266, device=pretrain_gene_x_batch.device).repeat(pretrain_gene_x_batch.shape[0], 1)

            value_labels = pretrain_gene_x_batch > 0
            x, x_padding = gatherData(pretrain_gene_x_batch, value_labels, pretrainconfig['pad_token_id'])

            position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
            
            x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
            position_emb = pretrainmodel.pos_emb(position_gene_ids)
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
            
            # Aggiungi gli embedding del batch alla lista complessiva
            all_gene_embeddings.append(geneembmerge.detach().cpu().numpy())

        # Pulizia della memoria dopo ogni batch
        del batch_data_subset
        del pretrain_gene_x_batch
        del data_gene_ids
        del value_labels
        del x, x_padding, position_gene_ids, position_emb, geneemb, geneemb1, geneemb2, geneemb3, geneemb4, geneembmerge
        torch.cuda.empty_cache() # Svuota la cache della GPU se stai usando CUDA
        
    # Concatena tutti gli embedding raccolti
    final_gene_embeddings = np.concatenate(all_gene_embeddings, axis=0)
    print(f"Generated total embeddings with shape: {final_gene_embeddings.shape}")
    np.save(strname, final_gene_embeddings)
    print("Inference complete and embeddings saved.")
