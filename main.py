
from src.preprocess import preprocess_h5ad
input_path = '//data2/home/vcivale/scTumorClassification2/dataset/f9ecb4ba-b033-4a93-b794-05e262dc1f59.h5ad'
gene_list_path = '/data2/home/vcivale/scTumorClassification2/model/scFoundation/OS_scRNA_gene_index.19264.tsv'
output_path = '/data2/home/vcivale/scTumorClassification2/dataset/neural.npz'
output_format = 'npz'
demo = False

preprocess_h5ad(input_path, gene_list_path, output_path, output_format, demo)


