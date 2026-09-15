"""Seeded Poisson synthetic request generator."""
import random
from scheduler.scheduler import Request

def generate(rate, count=30, seed=7):
    rng=random.Random(seed); now=0.; requests=[]
    for n in range(count):
        now += rng.expovariate(rate)
        requests.append(Request(str(n), rng.randint(8, 32), rng.randint(4, 12), 1 if rng.random()<.25 else 0, now, "shared" if n%4==0 else None))
    return requests
