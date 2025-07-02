import scanpy as sc
import numpy as np
import argparse
import scipy.sparse as sp
import pandas as pd
import scipy

def is_normalized(X):
    """Check se i dati sembrano normalizzati."""
    # Per matrici sparse converto in array per il check, ma solo piccola parte (sample)
    if sp.issparse(X):
        sample = X[:1000,:].toarray()  # campioniamo max 1000 righe per velocità
    else:
        sample = X[:1000,:]
    mean = sample.mean()
    min_val = sample.min()
    max_val = sample.max()
    return (0.5 < mean < 2) and (min_val >= 0) and (max_val < 50)


def main_gene_selection_sparse(matrix, genes_in_data, gene_list):
    """
    Seleziona e ordina colonne di matrix secondo gene_list,
    aggiungendo colonne zero per geni mancanti, senza DataFrame.

    matrix: scipy.sparse or np.ndarray (cells x genes)
    genes_in_data: list di geni corrispondenti a colonne di matrix
    gene_list: lista target di geni

    Ritorna:
    matrix_new: matrice cells x genes target (ordinata e padded)
    to_fill_columns: lista geni aggiunti con padding zero
    """

    genes_set = set(genes_in_data)
    gene_list_set = set(gene_list)
    to_fill_columns = list(gene_list_set - genes_set)
    common_genes = [g for g in gene_list if g in genes_set]

    # Indici colonne comuni
    common_indices = [genes_in_data.index(g) for g in common_genes]

    # Slicing matrice colonne comuni
    matrix_common = matrix[:, common_indices]

    # Crea matrice colonne zero per padding
    n_cells = matrix.shape[0]
    n_pad = len(to_fill_columns)
    if sp.issparse(matrix):
        zero_pad = sp.csr_matrix((n_cells, n_pad))
        matrix_new = sp.hstack([matrix_common, zero_pad], format='csr')
    else:
        zero_pad = np.zeros((n_cells, n_pad), dtype=matrix.dtype)
        matrix_new = np.hstack([matrix_common, zero_pad])


    return matrix_new, to_fill_columns


def preprocess_h5ad(input_path, gene_list_path, output_path, output_format='npz', demo=False):
    print(f"📥 Caricamento: {input_path}")
    adata = sc.read_h5ad(input_path)

    # Trova i nomi dei geni
    if 'Gene' in adata.var.columns:
        gene_names = adata.var['Gene'].tolist()
    elif 'original_gene_symbols' in adata.var.columns:
        gene_names = adata.var['original_gene_symbols'].tolist()
    else:
        raise ValueError("Gene column not found in adata.var. Expected 'Gene' or 'original_gene_symbols'.")

    # Controllo e normalizzazione
    if adata.raw is not None:
        print("⚠️ 'raw' presente in adata: assicurati che dati siano normalizzati.")
        matrix = adata.X
    else:
        matrix = adata.X
        if not is_normalized(matrix):
            print("⚠️ I dati NON sembrano normalizzati. Eseguo normalizzazione e log1p.")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            matrix = adata.X
        else:
            print("✅ I dati sembrano già normalizzati.")

    # Carica lista geni target
    gene_list_df = pd.read_csv(gene_list_path, sep='\t')
    target_gene_list = gene_list_df['gene_name'].tolist()

    # Quanti geni originari sono mantenuti?
    original_genes_set = set(gene_names)
    target_genes_set = set(target_gene_list)
    common_genes = original_genes_set.intersection(target_genes_set)
    print(f"🧬 Geni originari nel file: {len(original_genes_set)}")
    print(f"🧬 Geni target nella lista: {len(target_genes_set)}")
    print(f"✅ Geni comuni mantenuti: {len(common_genes)}")

    # Se demo, usiamo subset righe
    if demo:
        print("⚠️ Demo mode: uso solo prime 1000 cellule")
        matrix = matrix[:1000, :]

    # Selezione e padding senza DataFrame
    matrix_proc, padding_genes = main_gene_selection_sparse(matrix, gene_names, target_gene_list)

    # Salvataggio solo npz supportato
    if output_format != 'npz':
        raise ValueError("Attualmente supportato solo il formato 'npz'.")

    print(f"💾 Salvataggio in formato NPZ: {output_path}")
    scipy.sparse.save_npz(output_path, sp.csr_matrix(matrix_proc))
    print("✅ Completato!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessa un .h5ad e salva dati normalizzati in formato NPZ.")
    parser.add_argument('--input_path', required=True, help="Percorso del file .h5ad")
    parser.add_argument('--gene_list_path', default='./OS_scRNA_gene_index.19264.tsv', help="Lista geni target")
    parser.add_argument('--output_path', required=True, help="Dove salvare il file pre-elaborato")
    parser.add_argument('--output_format', choices=['npz'], default='npz', help="Formato di salvataggio")
    parser.add_argument('--demo', action='store_true', help="Esegui su un sottoinsieme per test")
    args = parser.parse_args()

    preprocess_h5ad(args.input_path, args.gene_list_path, args.output_path, args.output_format, args.demo)
