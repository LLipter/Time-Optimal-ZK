import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(8, 6.5))

    with open('se.txt', 'r') as fi:
        data = [line.strip().split() for line in fi.readlines()]
        xs = []
        part1 = []
        part2 = []
        part3 = []
        part4 = []
        ys = []
        for i in range(10):
            xs.append(int(data[i][0]))
            part1.append(float(data[i][1]))
            part2.append(float(data[i][2]))
            part3.append(float(data[i][3]))
            part4.append(float(data[i][4]))
            ys.append(float(data[i][5]))
        

    fig, ax = plt.subplots(1, 1)
    
    
    

    line2, = ax.plot(xs, part1, marker='*', color='darkblue', linewidth=3, alpha=0.7, label=r'$\frac{2}{q} + \frac{q-2}{q}(1 - \delta)^\lambda$')
    # line3, = ax.plot(xs, part2, marker='^', color='darkred', linewidth=1.5, alpha=0.7, label=r'$(1 - \delta)^\lambda$')
    line4, = ax.plot(xs, part3, marker='s', color='darkorange', linewidth=1.5, alpha=0.7, label=r'$\frac{2}{q-1} + \frac{q-3}{q-1}(1 - \frac{29\delta}{30})^\lambda$')
    line5, = ax.plot(xs, part4, marker='^', color='darkred', linewidth=1.5, alpha=0.7, label=r'$(1 - \frac{7\delta}{10})^\lambda$')
    line1, = ax.plot(xs, ys, marker='.', color='black', linewidth=1.5, alpha=0.7, label='Soundness Error')
    plt.legend(handles=[line1, line2, line4, line5], fontsize=13)

    ax.set_xscale(matplotlib.scale.LogScale(ax, base=2))
    ax.set_yscale(matplotlib.scale.LogScale(ax, base=2))
    ax.set_ylim(bottom=2**-280)
    plt.xticks(xs, [r'$2^{' + str(i) + r'}$' for i in range(7, 17)], fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title("Intel® Core™ i7-7700HQ CPU @ 2.80GHz (Kabylake)\nL1: 128KB, L2: 256KB, L3: 6MB\nCode Length=128", loc='left', fontsize=16, fontweight=1, color='black')
    plt.xlabel(r'Testing Tuples ($\lambda$)', fontsize=16)
    plt.ylabel("Soundness Error", fontsize=16)

    fig.savefig('se.pdf')