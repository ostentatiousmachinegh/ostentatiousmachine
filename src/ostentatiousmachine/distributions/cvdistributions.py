import secrets
import math


def uniform(a: float=0.0,b: float=1.0)->float:
    u=secrets.randbits(53)/(1<<53)
    return a+(b-a)*u


def exponentialdist(lam: float)->float:
    if lam<=0:
        raise ValueError("lambda must be positive")

    u=uniform(0.0,1.0)
    while u==0.0:
        u=uniform(0.0,1.0)
    return -math.log(u)/lam
