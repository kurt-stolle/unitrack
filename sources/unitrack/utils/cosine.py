import torch


def _stable_norm(t: torch.Tensor):
    return t / torch.linalg.vector_norm(t, dim=1, keepdim=True).clamp(min=1e-8)


@torch.jit.script
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Manual computation of the cosine similarity

    Based on: https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re  # noqa: E501

    ```
    csim(a,b) = dot(a, b) / (norm(a) * norm(b))
              = dot(a / norm(a), b / norm(b))
    ```

    Clamp with min(eps) for numerical stability. The final dot
    product is computed via transposed matrix multiplication (see `torch.mm`).
    """
    a_norm = _stable_norm(a)
    b_norm = _stable_norm(b)

    return torch.mm(a_norm, b_norm.T)


@torch.jit.script
def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance, i.e. a distance metric based on cosine similarity.
    """
    return 1 - cosine_similarity(a, b)
