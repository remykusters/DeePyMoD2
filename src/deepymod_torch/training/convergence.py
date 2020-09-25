"""This module implements convergence criteria 
"""

import torch


class Convergence:

    def __init__(self, patience: int = 100, delta: float = 0.05) -> None:
        """Implements convergence criterium. Convergence is when change in patience
        epochs is smaller than delta.

        Args:
            patience (int, optional): Number of epochs the L1 norm needs to stay constant to be considered converged. Defaults to 100.
            delta (float, optional): Maximum change of the L1 norm to be considered constant. Defaults to 0.05.
        """
        self.patience = patience
        self.delta = delta
        self.counter: int = 0
        self.start_l1: torch.Tensor = None
        self.converged = False

    def __call__(self, epoch: int, l1_norm: torch.Tensor) -> None:
        """ Decides if the DeepMoD sparsity selection is converged.  

        Args:
            epoch (int): Epoch at function call.
            l1_norm (torch.Tensor): Value of the L1 norm. 
        """
        if self.start_l1 is None:
            self.start_l1 = l1_norm
        elif torch.abs(self.start_l1 - l1_norm).item() < self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.converged = True
        else:
            self.start_l1 = l1_norm
            self.counter = 0
