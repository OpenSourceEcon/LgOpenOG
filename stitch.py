import numpy as np

def stitch(f_prime, domain, epsilon):
    def wrapper(f):
        lb, ub = domain
        if ub is None:
            lb = lb + epsilon
            def f_stitch(x, *args):
                if np.ndim(x) == 0:
                    if x < lb:
                        rv = f(lb, *args) + f_prime(lb, *args) * (x - lb)
                    else:
                        rv = f(x, *args)
                else:
                    rv = np.empty(len(x))
                    mask = x < lb
                    rv[mask] = f(lb, *args) + f_prime(lb, *args) * (x[mask] - lb)
                    rv[~mask] = f(x[~mask], *args)
                return rv
        elif lb is None:
            ub = ub - epsilon
            def f_stitch(x, *args):
                if np.ndim(x) == 0:
                    if x > ub:
                        rv = f(ub, *args) + f_prime(ub, *args) * (x - ub)
                    else:
                        rv = f(x, *args)
                else:
                    rv = np.empty(len(x))
                    mask = x > ub
                    rv[mask] = f(ub, *args) + f_prime(ub, *args) * (x[mask] - ub)
                    rv[~mask] = f(x[~mask], *args)
                return rv
        else:
            lb = lb + epsilon
            ub = ub - epsilon
            def f_stitch(x, *args):
                if np.ndim(x) == 0:
                    if x < lb:
                        rv = f(lb, *args) + f_prime(lb, *args) * (x - lb)
                    elif x > ub:
                        rv = f(ub, *args) + f_prime(ub, *args) * (x - ub)
                    else:
                        rv = f(x, *args)
                else:
                    rv = np.empty(len(x))
                    low_mask = x < lb
                    high_mask = x > ub
                    mid_mask = np.logical_and(~low_mask, ~high_mask)
                    rv[low_mask] = f(lb, *args) + f_prime(lb, *args) * (x[low_mask] - lb)
                    rv[high_mask] = f(ub, *args) + f_prime(ub, *args) * (x[high_mask] - ub)
                    rv[mid_mask] = f(x[mid_mask], *args)
                return rv
        return f_stitch
    return wrapper