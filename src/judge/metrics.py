from typing import Dict, Any

def default_metrics_schema() -> Dict[str, Any]:
    return {
        "revealness": None,
        "mapping_quality": None,
        "clarity": None,
        "creativity": None,
        "plausibility": None,
        "explanatory_power": None,
    }
