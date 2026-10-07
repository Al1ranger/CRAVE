import copy
from crave.common import digest

def checkpoint(state):
    payload = state.as_dict()
    return {"payload": copy.deepcopy(payload), "hash": digest(payload)}

def restore(saved):
    from crave.agent.state import State
    if digest(saved["payload"]) != saved["hash"]:
        raise ValueError("Checkpoint checksum mismatch")
    result = State.from_dict(copy.deepcopy(saved["payload"]))
    if digest(result.as_dict()) != saved["hash"]:
        raise ValueError("Checkpoint restore mismatch")
    return result
