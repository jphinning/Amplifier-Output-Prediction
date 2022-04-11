import matplotlib.pyplot as plt


def get_scattered_chart(input, output, c_title='Data', c_label='data', measured_input=[], measured_output=[]):
    # Ploting charts
    fig, ax = plt.subplots()

    if(len(measured_input) and len(measured_output)):
        plt.plot(measured_input, measured_output, 'o',
                 label="Measured Values", markersize=0.5)

    plt.plot(input, output, 'o',
             label=c_label, markersize=0.5)

    ax.set(xlabel='input', ylabel='output', title=c_title)
    plt.legend()
    ax.grid()

    fig.savefig(f"{c_title}.png")
