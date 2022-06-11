import math

def soundness_error(distance, test_no, t, code_len, field_size):
    d = code_len * distance
    part1 = d * (math.pow(d, t) - 1) / (4 * (d - 1) * field_size)
    part2 = math.pow(1 - min(math.pow(distance, t) / 4, 1 / 4), test_no)
    part3 = math.pow(1 - math.pow(distance, t) / 2, test_no)
    # print(part1)
    # print(part2)
    # print(part3)
    return part1 + part2 + part3


if __name__ == '__main__':
    fs = 46242760681095663677370860714659204618859642560429202607213929836750194081793
    print(soundness_error(0.07, 100, 2, 1762, fs))
    print(soundness_error(0.07, 1430, 3, 174, fs))
    print(soundness_error(0.07, 20400, 4, 56, fs))
    print(soundness_error(0.20, 100, 2, 1762, fs))
    print(soundness_error(0.20, 505, 3, 174, fs))
    print(soundness_error(0.20, 2550, 4, 56, fs))
    print(soundness_error(0.30, 100, 2, 1762, fs))
    print(soundness_error(0.30, 330, 3, 174, fs))
    print(soundness_error(0.30, 1100, 4, 56, fs))
    print(soundness_error(0.40, 100, 2, 1762, fs))
    print(soundness_error(0.40, 253, 3, 174, fs))
    print(soundness_error(0.40, 636, 4, 56, fs))
    print(soundness_error(0.50, 100, 2, 1762, fs))
    print(soundness_error(0.50, 204, 3, 174, fs))
    print(soundness_error(0.50, 411, 4, 56, fs))