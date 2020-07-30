import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from collections import namedtuple
import datetime

START_DATE = datetime.date(2020, 2, 24)
# TODAY = datetime.datetime.now().date()


Plot = namedtuple("Plot", ("x_label", "y_label", "use_grid", "use_legend", "curves", "bottom_adjust", "margins", "formatter", "h_pos", "v_pos"))
Plot.__new__.__defaults__ = (None,) * len(Plot._fields)

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
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    # plt.show()


def format_xtick(n, v):
    return (START_DATE + datetime.timedelta(int(n))).strftime("%d %b")  # e.g. "24 Feb", "25 Feb", ...


def generate_format_xtick(start_date):
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    def custom_xtick(n,v):
        return (start_date + datetime.timedelta(int(n))).strftime("%d %b")

    return custom_xtick


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
    plt.savefig(save_path, bbox_inches='tight', transparent=True)  # os.path.join(exp_path, exp_prefix + "sliding_SIR_global.pdf")
    plt.close('all')


def generic_plot(xy_curves, title, save_path, x_label=None, y_label=None, formatter=None, use_legend=True, use_grid=True, close=True, grid_spacing=20, yaxis_sci=False):
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
        ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_xtick))

    ax.xaxis.set_major_locator(MultipleLocator(grid_spacing))

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    ax.margins(0.05)
    if use_legend:
        ax.legend()

    if yaxis_sci:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=None)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

    if close:
        plt.close('all')
    return fig

def generic_sub_plot(subplots, title, save_path):
    """

    :param subplots:
    :param title:
    :param save_path:
    :return:
    """

    n_subplots = len(subplots)
    fig, axarr = plt.subplots(n_subplots, sharex=True, sharey=False, figsize=(9, 9))
    fig.suptitle(title)

    # ax = fig.add_subplot(n_subplots, sub_plot.h_pos, sub_plot.v_pos)
    for i,sub_plot in enumerate(subplots):
        for curve in sub_plot.curves:
            if curve.color is not None:
                axarr[i].plot(curve.x, curve.y, curve.style, label=curve.label, color=curve.color)
            else:
                axarr[i].plot(curve.x, curve.y, curve.style, label=curve.label)

            if sub_plot.formatter is not None:
                axarr[i].xaxis.set_major_formatter(plt.FuncFormatter(format_xtick))

        if sub_plot.use_legend:
            axarr[i].legend()

        if sub_plot.margins is not None:
            axarr[i].margins(sub_plot.margins)

        if sub_plot.y_label is not None:
            plt.ylabel(sub_plot.y_label)

    fig.subplots_adjust(hspace=0)
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axarr:
        ax.label_outer()

    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.close('all')
    return fig
