def performance_report(y_true, y_pred):
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred):
        if int(yt) == 1 and int(yp) == 1: tp += 1
        elif int(yt) == 0 and int(yp) == 1: fp += 1
        elif int(yt) == 1 and int(yp) == 0: fn += 1
        else: tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    }


def majority_vote(id3_pred, c45_pred, rf_pred): 
    votes = [int(id3_pred), int(c45_pred), int(rf_pred)]
    tally = {0: votes.count(0), 1: votes.count(1)}
    winner = max(tally, key=tally.get)

    return {
        "verdict": winner,
        "unanimous": len(set(votes)) == 1,
        "votes": {"id3": votes[0], "c45": votes[1], "random_forest": votes[2]},
        "tally": tally,
    }