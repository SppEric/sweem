import torch

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
	risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
	#print(pred)
	diff = pred - torch.log(risk_set_sum)
	yevent = yevent.cuda().float().reshape(-1, 1)
	#print(yevent.shape)
	sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
	cost = (- (torch.sum(sum_diff_in_observed / n_observed))).reshape((-1,))
	
	return(cost)

def temp_loss(pred, ytime, yevent):
    """
    Calculate the negative partial log likelihood for survival analysis.
    
    Parameters:
    pred: Predicted risk scores, Tensor of shape (batch_size,)
    ytime: Survival times, Tensor of shape (batch_size,)
    yevent: Event indicators (1 if event occurred, 0 otherwise), Tensor of shape (batch_size,)
    
    Returns:
    Cost: Calculated negative partial log likelihood.
    """
	print(ytime)
	print(yevent)
	
	n_sample = ytime.size(0)
    yevent = yevent.view(-1)
    pred = pred.view(-1)

    # Create lower triangular matrix of ones (Risk set matrix)
    matrix_ones = torch.ones(n_sample, n_sample, device=pred.device)
    ytime_indicator = torch.tril(matrix_ones)

    # Calculate risk set sum
    risk_set_sum = ytime_indicator.mm(torch.exp(pred))

    # Calculate the log likelihood
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    n_observed = yevent.sum(0)
    cost = -(sum_diff_in_observed / n_observed).reshape((-1,))

    return cost

# def temp_loss(risk_scores, times, events):
#     """
#     Calculate the custom loss function over a batch of data.
    
#     Parameters:
#     risk_scores: Tensor of shape (batch_size,)
#     times: Tensor of shape (batch_size,)
#     events: Tensor of shape (batch_size,)
    
#     Returns:
#     Loss value computed over the batch.
#     """
#     risk_scores = risk_scores.view(-1).cuda()
#     times = times.view(-1).cuda()
#     events = events.view(-1)
#     N = risk_scores.size(0)

#     # Repeat risk_scores and times to create a matrix for vectorized computation
#     risk_scores_matrix = risk_scores.repeat(N, 1).cuda()
#     times_matrix = times.repeat(N, 1).cuda()

#     # Mask for selecting valid pairs
#     valid_pairs_mask = torch.ge(times_matrix, times.view(-1, 1)).cuda()
#     # Mask for ignoring self-pairs
#     self_pairs_mask = ~torch.eye(N, dtype=torch.bool, device=risk_scores.device).cuda()
#     final_mask = (valid_pairs_mask & self_pairs_mask).cuda()

#     # Calculate the inner sum in a vectorized way
#     exp_risk_scores = torch.exp(risk_scores).cuda()
#     inner_sum = torch.where(final_mask, exp_risk_scores, torch.zeros_like(exp_risk_scores)).sum(dim=1).cuda()

#     # Calculate the overall loss
#     valid_events_mask = (events > 0).cuda()
#     log_inner_sum = torch.log(inner_sum).cuda()
#     overall_sum = torch.where(valid_events_mask, risk_scores - log_inner_sum, torch.zeros_like(risk_scores).cuda()).sum().cuda()

#     return -overall_sum / N