import sys
path = sys.argv[1]
print("exe time for " + path)
print("	full inf	store		load		partial inf")
with open(path, "r") as f:
	inf = 0
	store = 0
	load = 0
	par = 0
	count = 0
	for line in f:
		if line and line.startswith("d"):
			line = line.split("=")
			dist = int(line[1])
		elif line and line.startswith("f"):
			line = line.split(" ")
			line = [x[:-2] for x in line]
			inf += float(line[2])
			store += float(line[4])
			load += float(line[6])
			par += float(line[9])
			count += 1
			if count == dist:
				print("	%.4f, 	%.4f, 	%.4f, 	%.4f" % (inf/count, store/count, load/count, par/count))
				count = 0
				inf = 0
				store = 0
				load = 0
				par = 0
			
