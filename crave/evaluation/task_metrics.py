import re
import string
from collections import Counter

def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def answer_metrics(prediction, gold, aliases=()):
    prediction = normalize(prediction)
    best_em = best_f1 = 0.0
    for answer in [gold, *aliases]:
        answer = normalize(answer)
        em = float(prediction == answer)
        if prediction in ["yes", "no", "noanswer"] or answer in ["yes", "no", "noanswer"]:
            f1 = em
        else:
            p, g = prediction.split(), answer.split()
            common = sum((Counter(p) & Counter(g)).values())
            f1 = 2 * common / (len(p) + len(g)) if common else em
        best_em, best_f1 = max(best_em, em), max(best_f1, f1)
    return {"exact_match": best_em, "f1": best_f1, "task_score": best_em}

def future_cost(before, after):
    return [before.budget.tokens - after.budget.tokens,
            before.budget.retrievals - after.budget.retrievals,
            before.budget.verifications - after.budget.verifications,
            after.latency - before.latency]

def utility(reward, cost, weight, references=(4096, 4, 2, 60), include_latency=True):
    n = 4 if include_latency else 3
    return float(reward - weight * sum(cost[j] / references[j] for j in range(n)))
