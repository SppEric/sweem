def concordance_index(risk_scores, times):
    """
    Calculate the concordance index given risk scores and times
    :param risk_scores: risk scores for each patient
    :param times: survival times for each patient
    :return: concordance index
    """
    concordant_pairs = 0
    total_pairs = 0
    for p1_risk, p1_time in zip(risk_scores, times):
        for p2_risk, p2_time in zip(risk_scores, times):
            if p2_time < p1_time:
                total_pairs += 1
                if p2_risk > p1_risk:
                    concordant_pairs += 1
    c_index = concordant_pairs / total_pairs
    return c_index

def brier_score(risk_scores, events):
    """
    Calculate the Brier score given risk scores and events
    :param risk_scores: risk scores for each patient
    :param events: events for each patient
    :return: Brier score
    """
    brier_score = 0
    for risk_score, event in zip(risk_scores, events):
        brier_score += (risk_score - event) ** 2
    brier_score = brier_score / len(risk_scores)
    return brier_score
