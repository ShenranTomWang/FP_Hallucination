from .minicheck_operator import MiniCheckOperator
from typing import Dict, List

class CREPEMiniCheckOperator(MiniCheckOperator):
    def check(self, dp: Dict[str, any],) -> Dict[str, any]:
        presuppositions = dp['model_detected_presuppositions']['presuppositions']
        passages = " ||| ".join(dp['passages'])
        pred_labels, _, _, _ = self.model.score(docs=[passages for _ in presuppositions], claims=presuppositions)
        dp['minicheck_results'] = pred_labels
        return dp