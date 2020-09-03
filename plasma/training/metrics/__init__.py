from .standard_metrics import acc_fn, fb_fn


__mapping__ = {
    "accuracy": acc_fn,
    "acc": acc_fn,
    "fb": fb_fn,
    "fb_score": fb_fn,
    "fb score": fb_fn,
}
