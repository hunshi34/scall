import sys
from anndata import AnnData
from tqdm import tqdm
import numpy as np
import scanpy as sc
import json
import random
import os
import torch
# 添加根目录到系统路径

current_directory = os.getcwd()
# print(f"当前运行目录: {current_directory}")
sys.path.append(os.path.abspath(os.path.join(current_directory,'...', 'model', 'scGPT_main')))

from scgpt.tokenizer import GeneVocab

model_dir="/public/home/cit_xiaominyu/single-cell/check_point/scgpt-human"
gene_col='feature_name'
max_length=300
batch_size=64
obs_to_save= None
device= "cuda"
use_fast_transformer= True
return_new_adata= False

data_dir = "/public/home/cit_xiaominyu/single-cell/data/original_data"
# 创建一个空 adata 对象
adatas=[]


# 遍历每个文件夹
for folder in os.listdir(data_dir):
    # 拼接文件路径
    file_path=data_dir+'/'+folder+'/partition_0.h5ad'
    temp_adata = sc.read_h5ad(file_path)
    # 合并 adata 对象
    adatas.append(temp_adata[:2000])

for adata in adatas:
    if gene_col == "index":
        adata.var["index"] = adata.var.index
    else:
        assert gene_col in adata.var
vocab_file = "/public/home/cit_xiaominyu/single-cell/check_point/scgpt-human/vocab.json"
# vocab_file="/home/xh/single-cell/model/scGPT-main/scgpt/tokenizer/default_gene_vocab.json"
with open(vocab_file) as f:
    init_vocab = json.load(f)
re_vocab = {value: key for key, value in init_vocab.items()}
# vocabulary
vocab = GeneVocab.from_file(vocab_file)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
vocab.set_default_index(vocab["<pad>"])
for i in range(len(adatas)):
    adatas[i].var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adatas[i].var[gene_col]
    ]
    adatas[i] = adatas[i][:, adatas[i].var["id_in_vocab"] >= 0]
print(adatas[0].X)

def make_pretrain_id_name_set(data, revocab):
    # return:a dict with gene's name and id and expression,which have their expression over 0.
    count_matrix = data.X.toarray()
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
    )
    gene_ids = np.array(adata.var["id_in_vocab"])
    finetune_data = []
    for idx in tqdm(range(len(data))):
        row = count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = gene_ids[nonzero_idx]
        if len(genes) > 300:
            selected_indices = random.sample(range(len(genes)), 300)
            genes = genes[selected_indices]
            values = values[selected_indices]
        gene_names = [revocab[gene] for gene in genes]
        cell_data = {
            'gene_ids': genes.tolist(),
            'gene_names': str(gene_names),
            'expressions': values.tolist()
        }
        for key in data[idx].var.keys():
            cell_data[key] = str(data[idx].var[key])
        finetune_data.append(cell_data)
    return finetune_data


def pretrain_data_in_order(data, revocab):
    # return:a dict with gene's name and id and expression,which have their expression in order.
    count_matrix = data.X.toarray()
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
    )
    gene_ids = np.array(adata.var["id_in_vocab"])
    pretrain_data = []
    for idx in tqdm(range(len(data))):
        expression = count_matrix[idx]
        gene = gene_ids
        num_groups = len(gene) // 300 + 1
        for group_idx in range(num_groups):
            # 获取当前组的基因索引
            start_idx = group_idx * 300
            end_idx = min((group_idx + 1) * 300, len(gene))
            group_genes = gene[start_idx:end_idx]
            group_expressions = expression[start_idx:end_idx]
            # 获取当前组的基因名称
            group_gene_names = [revocab[gene] for gene in group_genes]

            # 构建当前组的cell_data
            pretrain_data.append({
                'gene_ids': group_genes.tolist(),
                'gene_names': group_gene_names,
                'expressions': group_expressions.tolist()
            })
    return pretrain_data

save_dir = "/public/home/cit_xiaominyu/single-cell/data/sc_pretrain/"
for i, adata in enumerate(adatas):
    dataset = make_pretrain_id_name_set(adata[:1800], re_vocab)
    dataset_complement=pretrain_data_in_order(adata[1800:],re_vocab)
    combined_dataset = dataset + dataset_complement
    filename = os.path.join(save_dir, os.listdir(data_dir)[i] + '.json')
    with open(filename, 'w') as f:
        json.dump(combined_dataset, f)