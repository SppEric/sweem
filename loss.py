import torch
from metrics import *

def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return(indicator_matrix)

def neg_par_log_likelihood(pred, ytime, yevent):
    n_observed = yevent.sum(0)
    ytime_indicator = R_set(ytime)
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    # print("pred shape: ", pred.shape)
    # print("pred: ", pred)
    risk_set_sum = ytime_indicator.mm(torch.exp(pred)) + 1e-9
    # print("risk_set_sum: ", risk_set_sum)
    diff = pred - torch.log(risk_set_sum)
    # print("diff: ", risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    # print("sum_diff_in_observed: ", sum_diff_in_observed)
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))

    return(cost)

def temp_loss(pred, ytime, yevent):
    # log(1) = 0
    # log(0) = -inf
    
    # good c_index = 1
    # bad c_index = 0
    c_index = concordance_index(pred, yevent, ytime)
    c_index_loss = -torch.log(c_index + 0.0001)
    
    # good brier = 0
    # bad brier = 1
    brier = brier_score(pred, yevent)
    brier_loss = -torch.log((1-brier) + 0.0001)
    
    print("c_index: ", c_index)
    print("c_index_loss: ", c_index_loss)
    print("brier: ", brier)
    print("brier_loss: ", brier_loss)
    
    loss = c_index_loss + brier_loss
    
    return loss

def temp_loss_2(pred, ytime, yevent):
    """
    Calculate the negative partial log likelihood for survival analysis.
    
    Parameters:
    pred: Predicted risk scores, Tensor of shape (batch_size,)
    ytime: Survival times, Tensor of shape (batch_size,)
    yevent: Event indicators (1 if event occurred, 0 otherwise), Tensor of shape (batch_size,)
    
    Returns:
    Cost: Calculated negative partial log likelihood.
    """
    n_sample = ytime.size(0)
    # yevent = yevent.view(-1)
    # pred = pred.view(-1)

    # Create lower triangular matrix of ones (Risk set matrix)
    matrix_ones = torch.ones(n_sample, n_sample, device=pred.device)
    ytime_indicator = torch.tril(matrix_ones)

    # Calculate risk set sum
    risk_set_sum = ytime_indicator.mm(torch.exp(pred))

    # Calculate the log likelihood
    diff = pred - torch.log(risk_set_sum)
    # yevent = yevent.float().reshape(-1, 1)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    n_observed = yevent.sum(0)
    cost = -(sum_diff_in_observed / n_observed).reshape((-1,))

    return cost