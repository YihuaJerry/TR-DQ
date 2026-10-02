import pickle
from pathlib import Path

import torch

Rot = {}


def _householder_base(n, device, dtype=torch.float32):
    """Create an orthogonal matrix whose first column is uniformly distributed."""

    first = torch.zeros(n, device=device, dtype=dtype)
    first[0] = 1
    target = torch.full((n,), n**-0.5, device=device, dtype=dtype)
    direction = first - target
    direction = direction / direction.norm()
    identity = torch.eye(n, device=device, dtype=dtype)
    return identity - 2 * torch.outer(direction, direction)


def get_rot(n, device='cpu'):
    """Return the randomized orthogonal rotation used by TR-DQ.

    Older experiment snapshots loaded a working-directory-relative ``Rot.pkl``
    that was not shipped. If a compatible cache exists beside this module it is
    still honored; otherwise the same required base property is constructed
    analytically with a Householder transform.
    """

    if n < 2:
        raise ValueError("rotation size must be at least 2")
    if Rot.get(n) is None:
        cache_path = Path(__file__).with_name("Rot.pkl")
        if cache_path.is_file():
            with cache_path.open("rb") as stream:
                Rot[n] = pickle.load(stream)[n]
        else:
            Rot[n] = _householder_base(n, device="cpu")
    rotation = Rot[n].to(device)
    random_matrix = torch.randn(n - 1, n - 1, device=device)
    q, _ = torch.linalg.qr(random_matrix)
    block_q = torch.eye(n, device=device, dtype=q.dtype)
    block_q[1:, 1:] = q
    return rotation.to(block_q.dtype) @ block_q

def exchange_row_col(_tensor, i, j):
    tensor = _tensor.detach().clone()
    assert isinstance(tensor, torch.Tensor)
    indices_row = torch.arange(tensor.size(0), device=tensor.device)
    indices_row[i], indices_row[j] = indices_row[j].item(), indices_row[i].item()
    tensor = tensor[indices_row]

    indices_col = torch.arange(tensor.size(1), device=tensor.device)
    indices_col[i], indices_col[j] = indices_col[j].item(), indices_col[i].item()
    tensor = tensor[:, indices_col]
    return tensor
