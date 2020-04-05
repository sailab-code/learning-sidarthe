import matplotlib.pyplot as plt
import datetime

START_DATE = datetime.date(2020, 2, 24)
# TODAY = datetime.datetime.now().date()


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
    plt.savefig(save_path)
    # plt.show()