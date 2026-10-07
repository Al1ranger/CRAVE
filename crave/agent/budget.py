from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Cost:
    tokens: int = 0
    retrievals: int = 0
    verifications: int = 0
    latency: float = 0.0

    def __post_init__(self):
        if min(self.tokens, self.retrievals, self.verifications, self.latency) < 0:
            raise ValueError("Negative resource cost")

    def vector(self):
        return [self.tokens, self.retrievals, self.verifications, self.latency]

@dataclass
class Budget:
    tokens: int
    retrievals: int
    verifications: int
    latency: float | None = None

    def __post_init__(self):
        if min(self.tokens, self.retrievals, self.verifications) < 0 or (self.latency is not None and self.latency < 0):
            raise ValueError("Negative budget")

    def allows(self, cost):
        return (self.tokens >= cost.tokens and self.retrievals >= cost.retrievals
                and self.verifications >= cost.verifications
                and (self.latency is None or self.latency >= cost.latency))

    def consume(self, cost):
        if not self.allows(cost):
            raise ValueError("Hard resource budget exceeded")
        self.tokens -= cost.tokens
        self.retrievals -= cost.retrievals
        self.verifications -= cost.verifications
        if self.latency is not None:
            self.latency -= cost.latency

    def as_dict(self):
        return asdict(self)
