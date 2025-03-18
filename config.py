import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENOTYPE_MATRIX = torch.tensor([[0.5] * 10, [0.5] * 10], device=device)

# PARAMETRY OBSZARU (dla pozycji)
area_width = 100.0
area_height = 100.0
