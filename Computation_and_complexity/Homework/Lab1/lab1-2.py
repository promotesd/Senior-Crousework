from itertools import product
from typing import Hashable, Tuple, Set, Iterable
import random

Pair=Tuple[Hashable, Hashable]

def reflexive_transitive_closure(A:Iterable[Hashable], R:Set[Pair])->Set[Pair]:
    A=tuple(A)
    n=len(A)
    R=set(R)
    Rstar:Set[Pair]=set()

    for a in A:
        Rstar.add((a,a))

    for i in range(2,n+1):
        for seq in product(A, repeat=i):
            flag=True
            for j in range(i-1):
                if(seq[j],seq[j+1]) not in R:
                    flag=False
                    break
            if flag:
                Rstar.add((seq[0],seq[-1]))

    return Rstar

def rand_R(n:int, edge_prob:float, seed:int=42, include_self_in_R:bool=False):
    rand=random.Random(seed)
    A=list(range(n))
    R:Set[Pair]=set()
    for x in A:
        for y in A:
            if (x==y) and not include_self_in_R:
                continue
            if rand.random()<edge_prob:
                R.add((x,y))
    return A,R

if __name__=="__main__":
    n=9
    p=0.05

    A, R = rand_R(n, p)

    print("A =", A)
    print("R (random edges) =", sorted(R))

    Rstar= reflexive_transitive_closure(A, R)

    print("\nR* by enumeration =", sorted(Rstar))


