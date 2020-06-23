with open('out.txt', 'r') as f:
    _sum = 0
    for line in f:
        if line.startswith('c'):
            items = line.split(' ')
            _sum += float(items[-1])
        elif 'Batch' in line:
            if _sum != 0:
                print(_sum)
                _sum = 0
