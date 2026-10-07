from dataclasses import dataclass, field, asdict
from crave.agent.budget import Budget

ACTIONS = ("THINK", "RETRIEVE", "VERIFY", "STOP")

@dataclass
class State:
    task_id: str
    dataset: str
    split: str
    question: str
    budget: Budget
    initial_budget: dict
    history: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    candidate_answer: str = ""
    verification: str = ""
    confidence: float = 0.0
    raw_confidence: float = 0.0
    convergence: float = 0.0
    evidence_sufficiency: float = 0.0
    novelty: float = 0.0
    step: int = 0
    previous_action: str = "STOP"
    terminal: bool = False
    input_tokens: int = 0
    context_tokens_added: int = 0
    retrieved_documents: int = 0
    latency: float = 0.0
    controller_latency: float = 0.0
    routing_tokens: int = 0
    seed: int = 0
    action_log: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["budget"] = Budget(**data["budget"])
        return cls(**data)

@dataclass
class ActionResult:
    action: str
    state_before: str
    state_after: str
    token_cost: int
    retrieval_cost: int
    verification_cost: int
    latency: float
    candidate_answer: str
    confidence: float
    metadata: dict = field(default_factory=dict)
