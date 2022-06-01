#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import numpy as np


class EarlyStopping:
    """Early stopping criterion. When called, monitor loss and return boolean
    wether loss decreased over the past 'patience' steps.

    Args:
        patience (int, optional): number of steps before stopping in case of
            non decreasing loss. Defaults to 5.
        delta (int>=0, optional): minimum value between two steps for the loss
            to be considered decreasing. Defaults to 0.
    """

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_loss = np.inf
        self.early_stopping = False

    def __call__(self, loss):
        if loss + self.delta > self.min_loss:
            self.counter += 1
            if loss < self.min_loss:
                self.min_loss = loss
            if self.counter >= self.patience:
                self.early_stopping = True
        else:
            self.counter = 0
            self.min_loss = loss
        return self.early_stopping
