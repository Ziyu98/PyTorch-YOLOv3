import sys
path = sys.argv[1]
dist = int(sys.argv[2])
output_path = sys.argv[3]
with open(path, "r") as f:
    lines = f.read().split("\n")
    i = 0
    j = 0
    k = 0
    l = 0
    sum_store = 0
    sum_load = 0
    sum_part_inf = 0
    sum_inf = 0
    output_file = open(output_path, "a")
    for line in lines:
        if line and line.startswith("f"):
            # full inference
            items = line.split("=")
            times = items[1].split(":")
            temp_time = float(times[2])
            sum_inf += temp_time
            i += 1
        elif line and line.startswith("s"):
            items = line.split("=")
            times = items[1].split(":")
            temp_time = float(times[2])
            sum_store += temp_time
            j += 1
        elif line and line.startswith("l"):
            items = line.split("=")
            times = items[1].split(":")
            temp_time = float(times[2])
            sum_load += temp_time
            k += 1
        elif line and line.startswith("p"):
            items = line.split("=")
            times = items[1].split(":")
            temp_time = float(times[2])
            sum_part_inf += temp_time
            l += 1
    output_file.write("dist = " + str(dist + 1) + "\n")
    text = "full inference: " + str(sum_inf / i) +", store: " + str(sum_store / (j * dist)) + ", load: " + str(sum_load / k) + ", partial inf: " + str(sum_part_inf / l) + "\n"
    output_file.write(text)
    output_file.close()
    #print("full inference: %f, store: %f, load: %f, partial inf: %f\n" % (sum_inf / i, sum_store / (j * dist), sum_load / k, sum_part_inf / l))
