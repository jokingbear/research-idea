from bayes_opt import BayesianOptimization


def optimize_aug(d, target_fn, n_points=1000, n_iter=100, seed=None, exploration=3):
    opt = BayesianOptimization(target_fn, d, random_state=seed)

    opt.maximize(init_points=n_points, n_iter=n_iter, kappa=exploration)

    return opt
