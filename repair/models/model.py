from typing import Any, Dict, List


class Pipeline:
    def __init__(self):
        pass

    def __call__(self, example: Dict[str, Any], **kwargs) -> List[str]:
        raise NotImplementedError("This method must be implemented by a subclass.")
