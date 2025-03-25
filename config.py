import torch

# Removed device assignment here
GENOTYPE_MATRIX = torch.tensor([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]], device='cuda').T

# AREA PARAMETERS
area_width = 100.0
area_height = 100.0
size = 2
n = 4
device = 'cuda'


genotypes = torch.rand(size, n-1, 2, device=device)
last_gene_values = torch.randint(0, 2, (size, 1), device=device)
last_gene = last_gene_values.unsqueeze(2).expand(-1, -1, 2)
genotypes = torch.cat([genotypes, last_gene], dim=1)

# print(genotypes)
genes = genotypes[:, :-1].mean(-1).unsqueeze(0)
print(genotypes)
phenos = torch.matmul(genes, GENOTYPE_MATRIX)

'''

positions = torch.rand(size, 2, device=device) * torch.tensor(
            [area_width, area_height], device=device)

print(positions)'
'''
sigma = 1

optimal_genotype = torch.rand((1, n-1, 2), device=device)
print(optimal_genotype)
optimal_pheno = torch.matmul(optimal_genotype.mean(-1), GENOTYPE_MATRIX)
print('Fenotypy', phenos)
print('Fajny Geny', optimal_genotype)
print('Optimum', optimal_pheno)
phenos[:, 0] = optimal_pheno
phenos[:, 1] = -optimal_pheno
phenotype_dist = torch.norm(phenos - optimal_pheno, dim=-1)
phenotype_fitness = torch.exp(-phenotype_dist**2/(2* sigma**2))
print('Fitness', phenotype_fitness)
