import json
import time
from dataclasses import asdict
import numpy as np
from crave.common import digest
from crave.agent.budget import Budget, Cost
from crave.agent.state import State, ActionResult, ACTIONS
from crave.models.backend import ANSWER_SCHEMA

class Environment:
    def __init__(self, backend, retriever, encoder, think_tokens=256, verify_tokens=192, context_cap=1400, max_steps=12):
        self.backend, self.retriever, self.encoder = backend, retriever, encoder
        self.think_tokens, self.verify_tokens = think_tokens, verify_tokens
        self.context_cap, self.max_steps = context_cap, max_steps

    def initial(self, task, budget, seed):
        # Explicit whitelist: answer, aliases, supporting_facts never enter State.
        return State(task_id=task["task_id"], dataset=task["dataset"], split=task["split"], question=task["question"],
                     budget=Budget(**budget), initial_budget=dict(budget), seed=seed)

    def reservation(self, action):
        return {"THINK": Cost(tokens=self.think_tokens), "RETRIEVE": Cost(retrievals=1),
                "VERIFY": Cost(tokens=self.verify_tokens, verifications=1), "STOP": Cost()}[action]

    def feasible(self, state):
        if state.terminal: return []
        if state.budget.latency is not None:
            raise NotImplementedError("Hard latency constraints require a certified executor deadline; Tier A measures latency only")
        if state.step >= self.max_steps: return ["STOP"]
        return [a for a in ACTIONS if state.budget.allows(self.reservation(a))
                and (a != "VERIFY" or bool(state.candidate_answer))]

    def prompt_state(self, state):
        return json.dumps({"question": state.question, "recent_summaries": state.history[-3:],
                           "evidence": state.evidence, "current_answer": state.candidate_answer,
                           "verification": state.verification}, ensure_ascii=True)

    def action(self, state, action):
        if action not in self.feasible(state): raise ValueError("Infeasible action: " + action)
        action_start = time.perf_counter()
        physical_operation_latency = 0.0
        before = digest(state.as_dict())
        metadata = {}
        tokens = inputs = ret = ver = 0
        latency = 0.0
        if action == "STOP":
            state.terminal = True
        elif action == "RETRIEVE":
            # Search query is derived solely from currently observable state.
            query = state.question + " " + " ".join(state.history[-1:])
            docs, latency = self.retriever.search(query, 5)
            physical_operation_latency = latency
            old_text = " ".join(d["text"] for d in state.evidence)
            seen = {d["doc_id"] for d in state.evidence}
            admitted, added = [], 0
            for doc in docs:
                if doc["doc_id"] in seen: continue
                text = self.backend.truncate(doc["text"], max(0, 280))
                count = self.backend.count(text)
                if state.context_tokens_added + added + count > self.context_cap: continue
                admitted.append({"doc_id": doc["doc_id"], "title": doc["title"], "text": text})
                added += count
                seen.add(doc["doc_id"])
            if admitted:
                v = self.encoder.encode([old_text, " ".join(d["text"] for d in admitted)])
                state.novelty = float(np.clip(1 - v[0] @ v[1], 0, 1)) if old_text else 1.0
            else: state.novelty = 0.0
            state.evidence.extend(admitted)
            state.context_tokens_added += added
            state.retrieved_documents += len(docs)
            ret = 1
            metadata = {"query": query, "top_k": 5, "returned_documents": len(docs), "admitted_documents": len(admitted), "context_tokens_added": added}
        else:
            system = ("Answer the question using available facts. Return compact JSON only: answer (short answer span), "
                      "summary (one short public reasoning summary, no private chain of thought), confidence (0 to 1), "
                      "support (0 to 1, evidence support). Do not claim facts from evidence that is absent.")
            if action == "VERIFY":
                system = ("Independently verify the current candidate against available evidence. Correct it only if evidence warrants. "
                          "Return JSON: answer, summary (brief verdict and cited document titles), confidence (0 to 1), support (0 to 1).")
                ver = 1
            generation_seed = int(digest({"seed": state.seed, "step": state.step, "action": action})[:8], 16)
            operation_start = time.perf_counter()
            result = self.backend.generate(system, self.prompt_state(state), self.reservation(action).tokens, generation_seed, ANSWER_SCHEMA)
            physical_operation_latency = time.perf_counter() - operation_start
            tokens, inputs, latency = result.output_tokens, result.input_tokens, result.latency
            metadata = {"cache_hit": result.cache_hit, "valid_json": result.valid_json, "generation_seed": generation_seed}
            if result.valid_json:
                data = result.data
                summary = str(data.get("summary", ""))[:1000]
                previous = state.history[-1] if state.history else ""
                if summary and previous:
                    vectors = self.encoder.encode([previous, summary])
                    state.convergence = float(np.clip(vectors[0] @ vectors[1], 0, 1))
                state.history.append(summary)
                state.candidate_answer = str(data.get("answer", "")).strip()[:500]
                state.raw_confidence = float(np.clip(float(data.get("confidence", 0)), 0, 1))
                state.confidence = state.raw_confidence
                state.evidence_sufficiency = float(np.clip(float(data.get("support", 0)), 0, 1)) if state.evidence else 0.0
                if action == "VERIFY": state.verification = summary
        latency += max(0.0, time.perf_counter() - action_start - physical_operation_latency)
        cost = Cost(tokens, ret, ver, latency)
        state.budget.consume(cost)
        state.input_tokens += inputs
        state.latency += latency
        state.previous_action = action
        state.step += 1
        after = digest(state.as_dict())
        result = ActionResult(action, before, after, tokens, ret, ver, latency, state.candidate_answer, state.confidence, metadata)
        state.action_log.append(asdict(result))
        return result
