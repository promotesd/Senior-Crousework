import random
import argparse

class Counter:
    def __init__(self):
        self.comps=0

    def lt(self,a,b):
        self.comps+=1
        return a<b
    
    def le(self,a,b):
        self.comps+=1
        return a<=b

def bubble_sort(a, c:Counter):
    arr=a[:]
    n=len(arr)
    for i in range(n-1):
        for j in range(n-i-1):
            if c.lt(arr[j+1],arr[j]):
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr

def insertion_sort(a,c:Counter,count_fail_check=False):
    arr=a[:]
    n=len(arr)
    for i in range(1,n):
        key=arr[i]
        j=i-1
        while j>=0 and c.lt(key,arr[j]):
            arr[j+1]=arr[j]
            j-=1
        if count_fail_check and j>=0:
            c.comps+=1

        arr[j+1]=key
    return arr

def merge_sorting(a,c:Counter):
    def merge(left, right):
        i=j=0
        out=[]
        while i<len(left) and j<len(right):
            if c.le(left[i],right[j]):
                out.append(left[i])
                i+=1
            else:
                out.append(right[j])
                j+=1
        if i<len(left):
            out.extend(left[i:])
        if j<len(right):
            out.extend(right[j:])

        return out
    n=len(a)
    if n<=1:
        return a[:]
    mid=n//2
    left=merge_sorting(a[:mid],c)
    right=merge_sorting(a[mid:],c)
    return merge(left, right)


def print_result(name, n, before, after, comps):
    print(f"=={name}==")
    print(f"n={n}")
    if n <= 20:
        print("The array before sorting:", " ".join(map(str, before)))
        print("The sorted array:", " ".join(map(str, after)))
    print(f"Number of algorithm comparisons: {comps}\n")


def main():
    arg=argparse.ArgumentParser(description="Sorting algorithm")
    arg.add_argument("--algorithm", choices=["bubble", "insertion", "merge", "all"],
                    default="all", help="which algorithm to run")
    arg.add_argument("--n", type=int, default=1000, help="array size")
    arg.add_argument("--seed", type=int, default=42, help="random seed")
    arg.add_argument("--mint", type=int, default=0, help="random min (inclusive)")
    arg.add_argument("--maxt", type=int, default=2000, help="random max (inclusive)")
    arg.add_argument("--insertion-failed-check",default=True, action="store_true",
                    help="count insertion's final failed comparison")
    args = arg.parse_args()

    rand=random.Random(args.seed)
    arr=[rand.randint(args.mint, args.maxt) for _ in range(args.n)]
    
    if args.algorithm in ("bubble", "all"):
        c = Counter()
        out = bubble_sort(arr, c)
        assert out == sorted(arr)
        print_result("Bubble Sort", args.n, arr, out, c.comps)

    if args.algorithm in ("insertion", "all"):
        c = Counter()
        out = insertion_sort(arr, c, count_fail_check=args.insertion_failed_check)
        assert out == sorted(arr)
        print_result("Insertion Sort", args.n, arr, out, c.comps)

    if args.algorithm in ("merge", "all"):
        c = Counter()
        out = merge_sorting(arr, c)
        assert out == sorted(arr)
        print_result("Merge Sort", args.n, arr, out, c.comps)

if __name__ == "__main__":
    main()

        
    