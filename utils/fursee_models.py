import torch


class FurseeModel(torch.nn.Module):
    def __init__(self, backbone, input_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.projection = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, embedding_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values, output_attentions=False, return_dict=True)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(pooled.float())
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
