import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(8, 6.5))

    with open('degree.txt', 'r') as fi:
        data = [line.strip() for line in fi.readlines()]
        x = []
        for i in range(9):
            x.append(0.5 - i * 0.05)
        degree = []
        for i in range(9):
            degree.append(int(data[i]))
        data = data[10:]
        time = []
        for i in range(9):
            time.append(int(data[i]))
        
    print(x)
    print(degree)
    print(time)

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    line1, = plt.plot(x, time, marker='.', color='black', linewidth=1.5, alpha=0.7, label='run time')
    # line2, = ax2.plot(x, degree, marker='*', color='darkblue', linewidth=1.5, alpha=0.7, label='degree')
    # line_mul3, = plt.plot(data_x, data_y3, marker='^', color='darkred', linewidth=1.5, alpha=0.7, label='add_inline_unfold_1x (n)')
    # line_mul4, = plt.plot(data_x, data_y4, marker='s', color='darkorange', linewidth=1.5, alpha=0.7, label='add_inline_unfold_1x_intrinsics (n)')
    # line_mul5, = plt.plot(data_x, data_y5, marker='8', color='cyan', linewidth=1.5, alpha=0.7, label='add_inline_unfold_2x_intrinsics (n)')
    # plt.legend(handles=[line1, line2], fontsize=8)

    # plt.xscale("log")
    # # plt.yscale("log")
    # plt.xticks(data_x, [r'$2^{' + str(i) + r'}$' for i in range(3, 23)], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.title("Intel® Core™ i7-7700HQ CPU @ 2.80GHz (Kabylake)\nL1: 128KB, L2: 256KB, L3: 6MB\nCode Length=128", loc='left', fontsize=16, fontweight=1, color='black')
    plt.xlabel("Relevant Distance", fontsize=16)
    plt.ylabel("Runtime[ms]", fontsize=16)

    plt.savefig('degree.pdf')

    # plt.show()