from dataclasses import asdict
import time
import numpy as np
from crave.agent.checkpoint import checkpoint, restore
from crave.agent.state import ACTIONS
from crave.common import digest
from crave.evaluation.task_metrics import answer_metrics, future_cost, utility

def rollout(env, state, policy, keep_states=False):
    trace = []
    while not state.terminal:
        saved = checkpoint(state)
        feasible = env.feasible(state)
        controller_before=state.controller_latency
        selection_start=time.perf_counter()
        action = policy(state, feasible)
        if state.controller_latency==controller_before:
            elapsed=time.perf_counter()-selection_start
            state.controller_latency+=elapsed
            state.latency+=elapsed
        # A router may consume tokens while selecting an action; recompute feasibility.
        if action not in env.feasible(state): raise ValueError("Policy returned infeasible action")
        result = env.action(state, action)
        if keep_states: trace.append({"checkpoint": saved, "action": action, "result": asdict(result)})
    return trace

def branch_seed(saved_hash, seed, replicate):
    return int(digest({"checkpoint": saved_hash, "seed": seed, "replicate": replicate})[:8], 16) % (2**31 - 1)

def branches(env, saved, task, continuation, k=3, seed=0, forced_sequences=None):
    before = restore(saved)
    sequences = forced_sequences if forced_sequences is not None else [(a,) for a in env.feasible(before)]
    records = []
    for sequence in sequences:
        for replicate in range(k):
            state = restore(saved)
            # Assert equality BEFORE assigning a randomized branch seed.
            if checkpoint(state)["hash"] != saved["hash"]: raise RuntimeError("FATAL branch mismatch")
            matched_payload = state.as_dict()
            state.seed = branch_seed(saved["hash"], seed, replicate)
            generation_seed = state.seed
            immediate = None
            valid = True
            for action in sequence:
                if action not in env.feasible(state):
                    valid = False
                    break
                env.action(state, action)
                if immediate is None:
                    immediate = {"reward": answer_metrics(state.candidate_answer, task["answer"], task.get("aliases", []))["task_score"],
                                 "cost": future_cost(before, state)}
            if not valid: break  # The ordered pair is not feasible under this intervention.
            if not state.terminal: rollout(env, state, continuation)
            metrics = answer_metrics(state.candidate_answer, task["answer"], task.get("aliases", []))
            records.append({"task_id": task["task_id"], "dataset": task["dataset"], "split": task["split"],
                            "state_id": saved["hash"][:16], "state_hash": saved["hash"], "state_before": matched_payload,
                            "budget": before.budget.as_dict(), "forced_sequence": list(sequence),
                            "forced_action": sequence[0] if sequence else "EMPTY", "replicate": replicate,
                            "branch_seed": generation_seed, "continuation_policy": continuation.version,
                            "reward": metrics["task_score"], "exact_match": metrics["exact_match"], "f1": metrics["f1"],
                            "future_cost": future_cost(before, state), "input_tokens": state.input_tokens-before.input_tokens,
                            "immediate": immediate, "final_answer": state.candidate_answer,
                            "action_log": state.action_log[len(before.action_log):]})
    return records

def interaction_targets(records, lambdas, references):
    # Pair labels have only the common feasible replicates and a common continuation regime.
    targets = {}
    grouped = {}
    for record in records: grouped.setdefault(tuple(record["forced_sequence"]), []).append(record)
    for pair, pair_rows in grouped.items():
        if len(pair) != 2: continue
        keys = [pair, (pair[0],), (pair[1],), ()]
        if any(key not in grouped for key in keys): continue
        common = set.intersection(*[{r["replicate"] for r in grouped[key]} for key in keys])
        if not common: continue
        for weight in lambdas:
            values = [np.mean([utility(r["reward"], r["future_cost"], weight, references) for r in grouped[key] if r["replicate"] in common]) for key in keys]
            targets[(pair, weight)] = float(values[0]-values[1]-values[2]+values[3])
    return targets
