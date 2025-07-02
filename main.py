from src.embedding import generate_cell_embeddings
import os

# Crea la directory 'results' se non esiste
output_dir = '/equilibrium/datasets/TCGA-histological-data//lung_embeddings'
os.makedirs(output_dir, exist_ok=True)

# Percorso al tuo file H5AD pre-processato
input_h5ad_file = '/equilibrium/datasets/TCGA-histological-data/lung_processed.h5ad' 
# Assicurati che questo file esista! Potresti doverlo creare con il tuo preprocess.py script.

# Se il modello non è in './models/models.ckpt', specifica il percorso esatto
model_ckpt_path = '/equilibrium/datasets/TCGA-histological-data/models1.ckpt'

print(f"Inizio generazione embedding per {input_h5ad_file}")

try:
    final_output_h5ad_path = generate_cell_embeddings(
        data_path=input_h5ad_file,
        save_path=output_dir,
        task_name='my_single_cell_analysis',
        input_type='singlecell',
        output_type='cell',
        pool_type='all',
        tgthighres='t4',
        pre_normalized='T', # Se i dati sono già normalizzati (dal preprocess.py), usa 'T'
        demo=False,          # Imposta a True se vuoi testare su un piccolo sottoinsieme (prime 10 cellule)
        version='ce',
        model_path=model_ckpt_path, # Decommenta e imposta se il modello non è nel percorso predefinito
        ckpt_name='01B-resolution',
        batch_size=1
    )
    print(f"\nProcesso completato. Embedding salvati in: {final_output_h5ad_path}")

except Exception as e:
    print(f"\nSi è verificato un errore durante la generazione degli embedding: {e}")