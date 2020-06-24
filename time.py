with open('out.txt', 'r') as f:
    _sum = 0
    _sum_p = 0
    for line in f:
        if line.startswith('f') and 'conv' in line:
            items = line.split(' ')
            print(float(items[-1]))
            _sum += float(items[-1])
        elif 'Batch' in line:
            if _sum != 0:
                print('******************sum of conv and padding:\n')
                print(_sum, _sum_p, '\n***********************next frame:\n')
                _sum = 0
                _sum_p = 0
        elif line.startswith('p') and 'conv' in line:
            items = line.split(' ')
            print(float(items[-1]))
            _sum += float(items[-1])
        elif line.startswith('p') and 'padding' in line:
            items = line.split(' ')
            print(float(items[-1]))
            _sum_p += float(items[-1])

