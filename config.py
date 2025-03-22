import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENOTYPE_MATRIX = torch.tensor([[3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]], device=device)

# PARAMETRY OBSZARU (dla pozycji)
area_width = 100.0
area_height = 100.0
