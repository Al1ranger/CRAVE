import time
import numpy as np
from crave.agent.budget import Cost
from crave.agent.state import ACTIONS
from crave.common import digest
from crave.models.backend import ROUTER_SCHEMA

class MixedPolicy:
    version = "mu0-v1"
    def __call__(self, state, feasible):
        # Fixed transparent bootstrap policy. No gold answers or outcomes.
        weights = {"THINK": 0.35, "RETRIEVE": 0.35, "VERIFY": 0.20, "STOP": 0.10}
        if not state.candidate_answer: weights["STOP"] = 0.02
        seed = int(digest({"seed": state.seed, "step": state.step, "policy": self.version})[:8], 16)
        rng = np.random.default_rng(seed)
        p = np.array([weights[a] for a in feasible]); p /= p.sum()
        return str(rng.choice(feasible, p=p))

class FixedPolicy:
    def __call__(self, state, feasible):
        schedule = ["RETRIEVE", "THINK", "VERIFY", "THINK", "STOP"]
        for action in schedule[min(state.step, 4):]:
            if action in feasible: return action
        return "STOP"

class ConfidencePolicy:
    def __init__(self, threshold=0.8): self.threshold = threshold
    def __call__(self, state, feasible):
        if state.candidate_answer and state.confidence >= self.threshold: return "STOP"
        if not state.evidence and "RETRIEVE" in feasible: return "RETRIEVE"
        if "THINK" in feasible: return "THINK"
        if "VERIFY" in feasible: return "VERIFY"
        return "STOP"

class LLMRouter:
    def __init__(self, env): self.env = env
    def __call__(self, state, feasible):
        # Reserve both router generation and any executable resource action.
        candidates = [a for a in feasible if state.budget.tokens >= 24 + self.env.reservation(a).tokens]
        if not candidates: return "STOP"
        result = self.env.backend.generate("Choose the most useful next action under the remaining budget. Return JSON with action only.",
                    self.env.prompt_state(state) + "\nRemaining budget: " + str(state.budget.as_dict()) + "\nAllowed: " + str(candidates),
                    24, int(digest({"seed": state.seed, "step": state.step, "policy": "llm-router"})[:8], 16), ROUTER_SCHEMA)
        state.budget.consume(Cost(tokens=result.output_tokens, latency=result.latency))
        state.routing_tokens += result.output_tokens
        state.input_tokens += result.input_tokens
        state.latency += result.latency
        state.controller_latency += result.latency
        action = result.data.get("action", "STOP")
        return action if action in candidates else "STOP"

class ValuePolicy:
    def __init__(self, model, features, weight, disable=()):
        self.model, self.features, self.weight, self.disable = model, features, weight, disable
    def values(self, state):
        start = time.perf_counter()
        outputs = self.model.predict(self.features(state, self.weight)[None])[0]
        elapsed=time.perf_counter() - start
        state.controller_latency += elapsed
        state.latency += elapsed
        return outputs[:24].reshape(4, 6)[:, 0]
    def __call__(self, state, feasible):
        values = self.values(state)
        allowed = [a for a in feasible if a not in self.disable]
        if not allowed: return "STOP"
        return max(allowed, key=lambda a: values[ACTIONS.index(a)])

class AggregatePolicy:
    version = "mu1-visitation-v1"
    def __init__(self, learned, epsilon=0.3): self.learned, self.epsilon = learned, epsilon
    def __call__(self, state, feasible):
        value = int(digest({"seed": state.seed, "step": state.step, "policy": self.version})[:8], 16) / (2**32)
        return MixedPolicy()(state, feasible) if value < self.epsilon else self.learned(state, feasible)

class PUMALikePolicy:
    """Proxy: summary cosine replaces the full windowed redundancy protocol."""
    def __call__(self, state, feasible):
        if state.convergence >= 0.95 and state.candidate_answer:
            if state.previous_action == "VERIFY" and state.evidence_sufficiency >= 0.8: return "STOP"
            if "VERIFY" in feasible: return "VERIFY"
        return "THINK" if "THINK" in feasible else "STOP"
