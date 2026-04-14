from typing import Dict


def heuristic_recovery_action(sample: Dict) -> str:
    """Heuristic that uses oracle failure_type (upper-bound reference)."""
    failure = sample.get("failure_type")
    attempted = sample.get("attempted_action", "")
    objects = sample.get("candidate_vocab", {}).get("objects", [])
    targets = sample.get("candidate_vocab", {}).get("targets", [])
    locations = sample.get("candidate_vocab", {}).get("locations", [])

    if failure == "F1":
        return "LookAround()"
    if failure == "F2":
        return f"Pick({objects[0]})" if objects else "LookAround()"
    if failure == "F3":
        if attempted.startswith("Pick(") and objects:
            return f"Navigate({objects[0]})"
        if attempted.startswith("Place(") and locations:
            return f"Navigate({locations[0]})"
        return "LookAround()"
    if failure == "F4":
        return "Retry(Pick)" if attempted.startswith("Pick(") else "LookAround()"
    if failure == "F5":
        if objects and locations:
            return f"Place({objects[0]},{locations[0]})"
        if targets:
            return f"Navigate({targets[0]})"
        return "LookAround()"

    return "LookAround()"


def blind_heuristic_recovery_action(sample: Dict) -> str:
    """Heuristic that does NOT use failure_type — fair comparison with VLM.

    Rules based only on attempted_action (which VLM also receives):
      - Navigate failed  →  LookAround()       (can't see destination)
      - Pick failed      →  Retry(Pick)         (interaction glitch)
      - Place failed     →  Retry(Place)
      - Open failed      →  Retry(Open)
      - Close failed     →  Retry(Close)
      - Otherwise        →  LookAround()
    """
    attempted = sample.get("attempted_action", "")

    if attempted.startswith("Navigate("):
        return "LookAround()"

    if attempted.startswith("Pick("):
        return "Retry(Pick)"

    if attempted.startswith("Place("):
        return "Retry(Place)"

    if attempted.startswith("Open("):
        return "Retry(Open)"

    if attempted.startswith("Close("):
        return "Retry(Close)"

    return "LookAround()"
