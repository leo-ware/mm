import torch
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from tqdm import tqdm

@dataclass
class OptimizationResult:
    loss: float
    params: Union[List[torch.Tensor], Dict[str, torch.Tensor]]
    optim_params: Union[Dict[str, Any], None] = None
    optimizer: Union[torch.optim.Optimizer, None] = None

def adam(params, loss_fn, max_epochs=1000, verbose=True, lr=0.01):
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in tqdm(range(max_epochs), disable=not verbose):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

    return OptimizationResult(
        loss=loss.item(),
        params=params,
        optimizer=optimizer,
        optim_params={
            'lr': lr,
            'max_epochs': max_epochs,
            'used_epochs': epoch,
        }
    )
