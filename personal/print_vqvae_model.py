import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mesh_vqvae', 'src'))
from mesh_vqvae.src.model import MaskedVQVAE3D
from mesh_vqvae.src.config import SmallModelConfig
import torch
from torchinfo import summary

model = MaskedVQVAE3D(SmallModelConfig())

# Create actual tensors with correct data types
batch_size = 2
num_points = 2048
num_query = 2048

inputs = [
    torch.randn(batch_size, num_points, 3),      # points: XYZ coordinates
    torch.randn(batch_size, num_points, 3),      # normals: Surface normals  
    torch.randn(batch_size, num_points, 1),      # curvature: Curvature values
    torch.randn(batch_size, num_query, 3),       # query_pts: Query points
    torch.randint(0, 40, (batch_size,))          # label: Class labels (int64)
]

# Show model summary with tensor shapes
summary(model, input_data=inputs, 
        col_names=["output_size", "num_params"],
        verbose=1)

