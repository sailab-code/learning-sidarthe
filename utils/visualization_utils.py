import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

from collections import namedtuple
import datetime

START_DATE = datetime.date(2020, 2, 24)
# TODAY = datetime.datetime.now().date()


Curve = namedtuple("Curve", ("x", "y", "style", "label", "color"))
Curve.__new__.__defaults__ = (None,) * len(Curve._fields)


def plot_data_and_fit(data, fitted_data, future_data, save_path, plot_name, curves_label="deceduti", x_scale=1.0):
    x, y = data
    x = [_v/x_scale for _v in x]
    fit_x, fit_y = fitted_data
    fit_x = [_v/x_scale for _v in fit_x]
    future_x, future_y = future_data
    future_x = [_v/x_scale for _v in future_x]

    def format_xtick(n, v):
        return (START_DATE + datetime.timedelta(int(n))).strftime("%d %b")  # e.g. "24 Feb", "25 Feb", ...

    fig, ax = plt.subplots()
    plt.title(plot_name)
    ax.plot(x, y, 'ro', label=curves_label)
    ax.plot(fit_x, fit_y, '-', color='lightblue', label=curves_label + " (fit)")
    ax.plot(future_x, future_y, '-', color='orange', label=curves_label + "(stimati)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xtick))
    ax.margins(0.05)
    plt.subplots_adjust(bottom=0.15)
    ax.legend()
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()


def format_xtick(n, v):
    return (START_DATE + datetime.timedelta(int(n))).strftime("%d %b")  # e.g. "24 Feb", "25 Feb", ...


def plot_sir_dynamic(s, i, r, region, save_path):
    """
    Plot SIR Dynamic
    :param s: RES[:,0]
    :param i: RES[:,1]
    :param r: RES[:,2]
    :param region:
    :param save_path: file to save the plot
    :return:
    """
    # Plotting
    fig = plt.figure(figsize=(9, 9))
    # pl.figure()

    ax = fig.add_subplot(3, 1, 1)
    # pl.subplot(311)
    plt.grid(True)
    plt.title('SIR  ({}'.format(region) + str(")"))
    pl_x = list(range(s.shape[0]))  # list(range(RES[:,0].shape[0]))
    ax.plot(pl_x, s, '-g', label='$x$')  # RES[:, 0]
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xtick))
    ax.margins(0.05)
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('S')
    plt.legend(loc=0)

    ax = fig.add_subplot(3, 1, 2)
    # pl.subplot(312)
    plt.grid(True)
    pl_x = list(range(i.shape[0]))  # RES[:,1]
    ax.plot(pl_x, i, '-r', label='$y$')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xtick))
    ax.margins(0.05)
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('I')
    plt.legend(loc=0)

    ax = fig.add_subplot(3, 1, 3)
    # pl.subplot(313)
    plt.grid(True)
    pl_x = list(range(r.shape[0]))  # RES[:,2]
    ax.plot(pl_x, r, '-k', label='$z$')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xtick))
    ax.margins(0.05)
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('R')
    plt.legend(loc=0)
    plt.savefig(save_path, bbox_inches='tight')  # os.path.join(exp_path, exp_prefix + "sliding_SIR_global.pdf")


def generic_plot(xy_curves, title, save_path, x_label=None, y_label=None, formatter=None, use_legend=True, use_grid=True):
    """

    :param xy_curves:
    :param title:
    :param x_label:
    :param y_label:
    :param formatter:
    :param save_path:
    :param use_legend:
    :param use_grid:
    :return:
    """

    fig, ax = plt.subplots()
    plt.title(title)
    plt.grid(use_grid)
    for curve in xy_curves:
        if curve.color is not None:
            ax.plot(curve.x, curve.y, curve.style, label=curve.label, color=curve.color)
        else:
            ax.plot(curve.x, curve.y, curve.style, label=curve.label)
    if formatter is not None:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xtick))

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    ax.margins(0.05)
    if use_legend:
        ax.legend()

    plt.savefig(save_path, bbox_inches='tight')
