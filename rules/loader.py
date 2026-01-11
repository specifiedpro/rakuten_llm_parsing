import importlib
import os
import yaml
from typing import Callable, List, Tuple

DEFAULT_REGISTRY = "rules/registry.yaml"

def load_registry(path: str = None) -> dict:
    reg_path = path or os.getenv("RULES_REGISTRY_PATH", DEFAULT_REGISTRY)
    if not os.path.exists(reg_path):
        return {"version": 1, "custom_rules": []}
    with open(reg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"version": 1, "custom_rules": []}

def load_custom_rule_functions(registry: dict) -> List[Tuple[int, Callable[[str], dict]]]:
    items = []
    for entry in registry.get("custom_rules", []) or []:
        if not entry.get("enabled", True):
            continue
        module_path = entry.get("module")
        func_name = entry.get("function")
        priority = int(entry.get("priority", 100))
        if not module_path or not func_name:
            continue
        try:
            mod = importlib.import_module(module_path)
            fn = getattr(mod, func_name)
            items.append((priority, fn))
        except Exception:
            continue
    items.sort(key=lambda x: x[0])  # lower number = higher priority
    return [(p, fn) for p, fn in items]