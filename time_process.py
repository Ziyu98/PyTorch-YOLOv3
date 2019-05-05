with open("time.txt", "r") as f:
    lines = f.read().split("\n")
    i = 0
    j = 0
    sum_store = 0
    sum_inf = 0
    for line in lines:
        if line:
            items = line.split(":")
            time = float(items[2].rstrip())
            if time > 1:
                sum_store += time
                i += 1
            else:
                sum_inf += time
                j += 1
    print(sum_store / i, sum_inf / j)
