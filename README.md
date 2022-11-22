# AMGDTI
AMGDTI: Drug-target interaction prediction based on adaptive meta-graph learning on heterogeneous network

# Model framework
![image](https://i.postimg.cc/4ywmGfKN/figure1.png)
# Requirements
- Pytorch
- sklearn
- scipy
- numpy
- transformers

# Quick start
To reproduce our results:
1.Run pretreatment.py to adjs_offset.pkl, neg_ratings_offset.npy, node_types.npy, pos_ratings_offset.npy
2.Run DTI.py to reproduce the cross validation results of AMGDTI with additional compound-protein binding affinity data. 

# Data description
- drug_smiles.csv: list of drug names and smiles.
- protein_fasta.csv: list of protein names and fasta.
- drug_se.dat : Drug-SideEffect association matrix.
- pro_pro.dat : Protein-Protein interaction matrix.
- drug_drug.dat : Drug-Drug interaction matrix.
- protein_dis.dat : Protein-Disease association matrix.
- drug_dis.dat : Drug-Disease association matrix.
- drug_target.dat : Drug-Protein interaction matrix.

These files: drug_smiles.csv, protein_fasta.csv, drug_se.dat, pro_pro.dat, drug_drug.dat, protein_dis.dat, drug_dis.dat, drug_target.dat from https://github.com/luoyunan/DTINet.
# Contacts
If you have any questions or comments, please feel free to email Zhiyang Hu (huzylife@163.com).
