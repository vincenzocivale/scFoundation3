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

# Assicurati che 'load.py' sia accessibile e contenga le funzioni necessarie.
# Ad esempio, potresti dover assicurarti che load.py si trovi nella stessa directory
# del tuo notebook o aggiungere il percorso al sys.path
try:
    from src.load import load_model_frommmf, gatherData, getEncoerDecoderData
except ImportError:
    print("Errore: Impossibile importare funzioni da 'load.py'. Assicurati che il file sia presente e accessibile.")
    # Potresti voler definire delle funzioni mock qui per il test,
    # o lanciare un errore più specifico. Per questo esempio, assumiamo sia disponibile.
    raise

# gene_list_df e gene_list: Carica la lista dei geni una volta.
# Potrebbe essere un parametro della funzione se la lista cambia frequentemente.
try:
    gene_list_df = pd.read_csv('/home/vcivale/scFoundation3/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
except FileNotFoundError:
    print("Errore: Il file 'OS_scRNA_gene_index.19264.tsv' non trovato.")
    print("Assicurati che la lista di geni sia disponibile. La funzione potrebbe non funzionare correttamente.")
    raise RuntimeError("La lista dei geni non è disponibile. Assicurati che il file esista e sia accessibile.")

import random
import os
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
import h5py

# Assicurati che 'load.py' sia accessibile e contenga le funzioni necessarie.
try:
    from src.load import load_model_frommmf, gatherData, getEncoerDecoderData
except ImportError:
    print("Errore: Impossibile importare funzioni da 'load.py'. Assicurati che il file sia presente e accessibile.")
    raise

# gene_list_df e gene_list: Carica la lista dei geni una volta.
try:
    gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
except FileNotFoundError:
    print("Errore: Il file 'OS_scRNA_gene_index.19264.tsv' non trovato.")
    print("Assicurati che la lista di geni sia disponibile. La funzione potrebbe non funzionare correttamente.")
    gene_list = None


def generate_cell_embeddings(
    data_path: str,
    save_path: str = './',
    task_name: str = 'deepcdr',
    input_type: str = 'singlecell',
    output_type: str = 'cell',
    pool_type: str = 'all',
    tgthighres: str = 't4',
    pre_normalized: str = 'T',
    demo: bool = False,
    version: str = 'ce',
    model_path: str = 'None',
    ckpt_name: str = '01B-resolution',
    batch_size: int = 256,
    seed: int = 0
) -> str:
    """
    Genera embedding di cellule da dati single-cell pre-processati e li salva incrementalmente
    in un file HDF5, associando gli embedding agli ID delle cellule.
    Questa versione è ottimizzata per la memoria, gestendo le matrici sparse batch per batch.

    Args:
        data_path (str): Percorso al file H5AD contenente i dati pre-processati.
        save_path (str): Directory dove salvare il file H5AD con gli embedding.
        task_name (str): Nome del task per il nome del file di output.
        input_type (str): Tipo di input ('singlecell' o 'bulk'). Default: 'singlecell'.
        output_type (str): Tipo di embedding da generare ('cell', 'gene', 'gene_batch', 'gene_expression').
                           Attualmente ottimizzato solo per 'cell'. Default: 'cell'.
        pool_type (str): Tipo di pooling per l'embedding della cellula ('all' o 'max').
                         Valido solo per output_type='cell'. Default: 'all'.
        tgthighres (str): Targeted high resolution (es. 't4', 'fX', 'aX'). Valido solo per 'singlecell'.
        pre_normalized (str): Indica se i dati sono già normalizzati ('F', 'T', 'A').
                              Si consiglia 'T' se preprocess.py è stato eseguito. Default: 'T'.
        demo (bool): Se True, elabora solo i primi 10 campioni per demo. Default: False.
        version (str): Versione del modello ('ce' per cell embedding, 'rde' per read depth enhancement).
                       Valido solo per output_type='cell'. Default: 'ce'.
        model_path (str): Percorso al modello pre-addestrato se version == 'noversion'. Default: 'None'.
        ckpt_name (str): Nome del checkpoint, usato nel nome del file di output. Default: '01B-resolution'.
        batch_size (int): Dimensione del batch per l'inferenza. Default: 256.
        seed (int): Seed per la riproducibilità. Default: 0.

    Returns:
        str: Il percorso al file H5AD finale con gli embedding in .obsm['X_cell_embedding'].

    Raises:
        ValueError: Se il formato di input non è H5AD, il numero di geni non corrisponde,
                    o i parametri non sono validi.
        FileNotFoundError: Se il file della lista dei geni o del modello non viene trovato.
    """
    if gene_list is None:
        raise RuntimeError("La lista dei geni 'OS_scRNA_gene_index.19264.tsv' non è stata caricata. Assicurati che il file esista.")

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {data_path}...")
    if not data_path.endswith('.h5ad'):
        raise ValueError("Input data must be an h5ad file for this optimized script.")
    
    adata = sc.read_h5ad(data_path)
    cell_ids = np.array(adata.obs_names.tolist()) # Converti a NumPy array per indicizzazione efficiente
    gexpr_feature = adata.X # Questa ora è la matrice sparsa (es. csr_matrix)

    # NON convertire l'intera matrice sparsa in densa qui per risparmiare memoria
    if not issparse(gexpr_feature):
        print("Warning: Input matrix is not sparse. This script is optimized for sparse input.")
    
    if gexpr_feature.shape[1] != 19264:
        raise ValueError(f"Number of genes ({gexpr_feature.shape[1]}) in input H5AD does not match expected 19264. Please ensure preprocess.py has been run correctly.")

    if demo:
        print("⚠️ Demo mode: processing only first 10 samples.")
        gexpr_feature = gexpr_feature[:10, :]
        cell_ids = cell_ids[:10]
    
    print(f"Loaded data shape: {gexpr_feature.shape}")
    print(f"Number of cells: {len(cell_ids)}")

    # Load model
    if version == 'noversion':
        key=None
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please provide a valid path or ensure it's in './models/'.")

        if output_type == 'cell':
            if version == 'ce':
                key = 'cell'
            elif version == 'rde':
                key = 'rde'
            else:
                raise ValueError(f'Invalid version for cell output_type: {version}. Expected "ce" or "rde".')
        elif output_type == 'gene':
            key = 'gene'
        elif output_type == 'gene_batch':
            key = 'gene'
        elif output_type == 'gene_expression':
            key = 'gene'
        else:
            raise ValueError(f'Invalid output_type: {output_type}. Must be one of cell, gene, gene_batch, gene_expression.')
    
    pretrainmodel, pretrainconfig = load_model_frommmf(model_path, key)
    pretrainmodel.eval()
    pretrainmodel.to(device)

    # Assicurati che la directory di salvataggio esista
    os.makedirs(save_path, exist_ok=True)
    
    output_h5ad_path = os.path.join(save_path, f"{task_name}_{ckpt_name}_{input_type}_{output_type}_embedding_{tgthighres}_resolution.h5ad")
    print(f'Output H5AD will be saved at {output_h5ad_path}')

    # Inizializza l'oggetto AnnData finale con i metadati originali
    adata_final = adata.copy() # Copia l'AnnData originale per aggiungervi gli embedding

    # Lista per tenere traccia dei percorsi dei singoli file HDF5 per batch
    batch_temp_files = []

    # Inference in batch
    num_cells = gexpr_feature.shape[0]
    num_batches = (num_cells + batch_size - 1) // batch_size

    if output_type == 'cell':
        # Ottieni una stima della dimensione dell'embedding da un singolo forward
        # Invece di zeros, usa rand per assicurare che dummy_value_labels non sia tutto False
        dummy_input = torch.rand(1, 19266).float().to(device) # Usa 19266, non gexpr_feature.shape[1] direttamente se il modello si aspetta 19266
        # A seconda della logica di gatherData, potresti voler aggiungere l'ultima colonna del totalcount
        # Se 19264 sono i geni e gli ultimi 2 sono totalcount/target_res
        if pre_normalized == 'A':
             # Se il modello si aspetta 19264 geni + 2 valori extra (totalcount, target_res)
            dummy_input = torch.rand(1, 19264 + 2).float().to(device)
        else: # per 'T' o 'F'
            dummy_input = torch.rand(1, 19264 + 2).float().to(device) # Assicurati che le dimensioni corrispondano a quelle che entrano nel modello!

        # Fai in modo che ci sia sempre almeno un valore "vero" per value_labels
        # Questo è cruciale se gatherData filtra basandosi su value_labels
        dummy_value_labels = dummy_input > 0 # Ora sarà quasi tutto True
        
        # Aggiusta il data_gene_ids per il dummy input se necessario.
        # Il modello sembra aspettarsi 19266 token.
        dummy_data_gene_ids = torch.arange(19266, device=device).repeat(1, 1)

        # Ora usa dummy_data_gene_ids nel gatherData per i position_gene_ids
        dummy_x, dummy_x_padding = gatherData(dummy_input, dummy_value_labels, pretrainconfig['pad_token_id'])
        
        dummy_position_gene_ids, _ = gatherData(dummy_data_gene_ids, dummy_value_labels, pretrainconfig['pad_token_id'])
        
        # Il resto del codice rimane uguale
        dummy_x = pretrainmodel.token_emb(torch.unsqueeze(dummy_x, 2).float(), output_weight = 0)
        dummy_position_emb = pretrainmodel.pos_emb(dummy_position_gene_ids)
        dummy_x += dummy_position_emb
        dummy_geneemb = pretrainmodel.encoder(dummy_x, dummy_x_padding)
        
        dummy_geneemb1 = dummy_geneemb[:,-1,:]
        dummy_geneemb2 = dummy_geneemb[:,-2,:]
        dummy_geneemb3, _ = torch.max(dummy_geneemb[:,:-2,:], dim=1)
        dummy_geneemb4 = torch.mean(dummy_geneemb[:,:-2,:], dim=1)
        if pool_type=='all':
            dummy_geneembmerge = torch.concat([dummy_geneemb1,dummy_geneemb2,dummy_geneemb3,dummy_geneemb4],axis=1)
        elif pool_type=='max':
            dummy_geneembmerge, _ = torch.max(dummy_geneemb, dim=1)
        
        embedding_dim = dummy_geneembmerge.shape[1]
        print(f"Detected embedding dimension: {embedding_dim}")

        for i_batch in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = i_batch * batch_size
            end_idx = min((i_batch + 1) * batch_size, num_cells)
            
            # Estrai la fetta sparsa del batch
            batch_gexpr_feature_sparse = gexpr_feature[start_idx:end_idx, :]
            batch_cell_ids = cell_ids[start_idx:end_idx] # NumPy array slice

            # CONVERTI IN DENSE SOLO PER IL BATCH CORRENTE
            if issparse(batch_gexpr_feature_sparse):
                batch_gexpr_feature_dense = batch_gexpr_feature_sparse.toarray()
            else:
                batch_gexpr_feature_dense = batch_gexpr_feature_sparse # Già denso

            with torch.no_grad():
                pretrain_gene_x_list = []
                # Preparazione dati per il batch (logica invariata rispetto a prima, ma usa _dense)
                if input_type == 'bulk':
                    for row_idx in range(batch_gexpr_feature_dense.shape[0]):
                        totalcount = np.sum(batch_gexpr_feature_dense[row_idx, :]) if pre_normalized == 'T' else np.log10(np.sum(batch_gexpr_feature_dense[row_idx, :]))
                        pretrain_gene_x_list.append(batch_gexpr_feature_dense[row_idx, :].tolist() + [totalcount, totalcount])
                
                elif input_type == 'singlecell':
                    for row_idx in range(batch_gexpr_feature_dense.shape[0]):
                        if pre_normalized == 'F':
                            total_sum = np.sum(batch_gexpr_feature_dense[row_idx, :])
                            tmpdata = (np.log1p(batch_gexpr_feature_dense[row_idx, :] / total_sum * 1e4)).tolist()
                        elif pre_normalized == 'T':
                            tmpdata = batch_gexpr_feature_dense[row_idx, :].tolist()
                        elif pre_normalized == 'A':
                            # Se pre_normalized è 'A', l'ultima colonna contiene il totalcount
                            tmpdata = batch_gexpr_feature_dense[row_idx, :-1].tolist() 
                        else:
                            raise ValueError('pre_normalized must be T,F or A')

                        totalcount = batch_gexpr_feature_dense[row_idx, -1] if pre_normalized == 'A' else np.sum(batch_gexpr_feature_dense[row_idx, :])

                        if tgthighres[0] == 'f':
                            pretrain_gene_x_list.append(tmpdata + [np.log10(totalcount * float(tgthighres[1:])), np.log10(totalcount)])
                        elif tgthighres[0] == 'a':
                            pretrain_gene_x_list.append(tmpdata + [np.log10(totalcount) + float(tgthighres[1:]), np.log10(totalcount)])
                        elif tgthighres[0] == 't':
                            pretrain_gene_x_list.append(tmpdata + [float(tgthighres[1:]), np.log10(totalcount)])
                        else:
                            raise ValueError('tgthighres must be start with f, a or t')
                else:
                    raise ValueError('input_type not supported')

                pretrain_gene_x = torch.tensor(pretrain_gene_x_list).float().to(device)
                data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

                value_labels = pretrain_gene_x > 0
                x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'])

                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
                position_emb = pretrainmodel.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = pretrainmodel.encoder(x,x_padding)

                geneemb1 = geneemb[:,-1,:]
                geneemb2 = geneemb[:,-2,:]
                geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
                geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
                if pool_type=='all':
                    geneembmerge = torch.concat([geneemb1,geneemb2,geneemb3,geneemb4],axis=1)
                elif pool_type=='max':
                    geneembmerge, _ = torch.max(geneemb, dim=1)
                else:
                    raise ValueError('pool_type must be all or max')
                
                batch_embeddings_np = geneembmerge.detach().cpu().numpy() # Gli embedding del batch
                
                # Salva il batch corrente in un file HDF5 temporaneo separato
                batch_temp_h5_file = os.path.join(save_path, f".temp_batch_{i_batch}_{os.getpid()}.h5")
                with h5py.File(batch_temp_h5_file, 'w') as f_batch:
                    f_batch.create_dataset('embeddings', data=batch_embeddings_np, compression="gzip")
                    dt = h5py.string_dtype(encoding='utf-8')
                    f_batch.create_dataset('cell_ids', data=batch_cell_ids.astype(dt), compression="gzip")
                    
                batch_temp_files.append(batch_temp_h5_file)
                # print(f"Saved batch {i_batch+1}/{num_batches} to {batch_temp_h5_file}") # Meno verboso in tqdm

    else: # Per output_type diversi da 'cell'
        print(f"Warning: '{output_type}' output type is not optimized for incremental saving and ID association in this function. No H5AD will be generated directly.")
        return None
    
    # Una volta completata l'inferenza di tutti i batch, unisci i risultati
    print("Combining temporary embedding files...")
    all_embeddings_list = []
    all_cell_ids_list = []

    for temp_file in tqdm(batch_temp_files, desc="Reading temp files"):
        with h5py.File(temp_file, 'r') as f_batch:
            all_embeddings_list.append(f_batch['embeddings'][()])
            all_cell_ids_list.append(f_batch['cell_ids'][()])
        os.remove(temp_file) # Pulisci il file temporaneo dopo la lettura

    final_embeddings_np = np.vstack(all_embeddings_list)
    final_cell_ids_np = np.concatenate(all_cell_ids_list)

    # Converti gli ID delle cellule da byte a stringhe Python
    final_cell_ids_str = [cid.decode('utf-8') for cid in final_cell_ids_np]

    print(f"Final combined embeddings shape: {final_embeddings_np.shape}")
    print(f"Final combined cell IDs count: {len(final_cell_ids_str)}")

    # Crea un DataFrame pandas per la reindicizzazione
    embeddings_df = pd.DataFrame(final_embeddings_np, index=final_cell_ids_str)
    embeddings_df.index.name = 'cell_id'

    # Assicurati che gli indici siano allineati prima di aggiungere gli embedding
    if not adata_final.obs_names.equals(embeddings_df.index):
        print("Warning: Cell IDs from original adata and generated embeddings do not match exactly. Attempting to reindex.")
        embeddings_df = embeddings_df.reindex(adata_final.obs_names).dropna()
        if len(embeddings_df) != len(adata_final.obs_names):
            print(f"Warning: After reindexing, {len(adata_final.obs_names) - len(embeddings_df)} cells might be missing embeddings. Check 'demo' mode or filtering.")

    adata_final.obsm['X_cell_embedding'] = embeddings_df.values
    
    adata_final.write(output_h5ad_path)
    print(f"✅ Cell embeddings saved to {output_h5ad_path} in .obsm['X_cell_embedding']")
    return output_h5ad_path