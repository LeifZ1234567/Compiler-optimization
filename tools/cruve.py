
import matplotlib.pyplot as plt


def generate_curve(name,curve):

    plt.title(name,fontsize = 12)
    plt.ylim((4.0,4.6))
    plt.ylabel("time",fontsize = 12)
    plt.xlabel("generation",fontsize = 12)

    ax = plt.gca()
    ax.xaxis.grid(True, which='major',linestyle = (0,(8,4))) 
    ax.yaxis.grid(True, which='major',linestyle = (0,(8,4))) 
    ax.plot(data,color = 'cornflowerblue',alpha = 0.7, linewidth=3,label='GA')
    # ax.axhline(y=13.538, c="r", ls="-.", lw=2, label='-O0:13.538')
    ax.axhline(y=5.334, c="steelblue", ls=":", lw=3, label='-O1:5.334')
    ax.axhline(y=4.471, c="steelblue", ls="--", lw=3, label='-O2:4.452')
    ax.axhline(y=4.412, c="steelblue", ls="-.", lw=3, label='-O3:4.412')
    ax.legend()
    plt.savefig("plot.jpg")


if __name__ == "__main__":

    #DE
    data = [4.373, 4.253, 4.253, 4.253, 4.253, 4.245, 4.185, 4.185, 4.148, 4.141, 4.078, 4.078, 4.078, 4.078, 4.068, 4.06, 4.055, 4.055, 4.055, 4.055, 4.055, 4.055, 4.055, 4.055, 4.055, 4.055, 4.055, 4.055, 4.055, 4.055]
    # JAYA
    data = [4.688, 4.337, 4.337, 4.259, 4.259, 4.259, 4.259, 4.259, 4.259, 4.146, 4.146, 4.101, 4.101, 4.047, 4.047, 4.047, 4.047, 4.047, 4.047, 4.047, 4.047, 4.047, 4.047, 4.047, 4.029, 4.029, 4.029, 4.029, 4.029, 4.029, 4.029]
    # GA
    data = [4.497, 4.353, 4.219, 4.219, 4.219, 4.219, 4.219, 4.204, 4.204, 4.204, 4.204, 4.204, 4.204, 4.204, 4.163, 4.163, 4.163, 4.163, 4.163, 4.163, 4.108, 4.091, 4.091, 4.091, 4.091, 4.091, 4.091, 4.091, 4.091, 4.091]

    plt.title("Differential Evolution Algorithm",fontsize = 12)
    plt.title("JAYA Algorithm",fontsize = 12)
    plt.title("Gengetic Algorithm",fontsize = 12)
    plt.ylim((4.0,4.6))
    plt.ylabel("time",fontsize = 12)
    plt.xlabel("generation",fontsize = 12)

    ax = plt.gca()
    ax.xaxis.grid(True, which='major',linestyle = (0,(8,4))) 
    ax.yaxis.grid(True, which='major',linestyle = (0,(8,4))) 


    # ax.plot(data,color = 'blue',alpha = 0.7, linewidth=2.3,label='DE')
    # ax.plot(data,color = 'blue',alpha = 0.7, linewidth=2.3,label='JAYA')
    ax.plot(data,color = 'cornflowerblue',alpha = 0.7, linewidth=3,label='GA')
    # ax.axhline(y=13.538, c="r", ls="-.", lw=2, label='-O0:13.538')
    ax.axhline(y=5.334, c="steelblue", ls=":", lw=3, label='-O1:5.334')
    ax.axhline(y=4.471, c="steelblue", ls="--", lw=3, label='-O2:4.452')
    ax.axhline(y=4.412, c="steelblue", ls="-.", lw=3, label='-O3:4.412')
    ax.legend()
    plt.savefig("plot.jpg")