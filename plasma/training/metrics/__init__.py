from .standard_metrics import Accuracy, FbetaScore


__mapping__ = {
    "accuracy": Accuracy,
    "acc": Accuracy,
    "fb": FbetaScore,
    "fb_score": FbetaScore,
    "fb score": FbetaScore,
}
