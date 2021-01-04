from datetime import datetime

def get_exp_prefix(area, initial_params, learning_rates, train_size, val_len, der_1st_reg,
                   t_inc, momentum, m, a, loss_type, integrator):
    prefix = f"{area[0]}_{integrator.__name__}"
    for key, value in initial_params.items():
        prefix += f"_{key[0]}{value}"

    for key, value in learning_rates.items():
        prefix += f"_{key[0]}{value}"

    prefix += f"_ts{train_size}_vs{val_len}_der1st{der_1st_reg}_tinc{t_inc}_momentum{momentum}_m{m}_a{a}_loss{loss_type}"

    prefix += f"{datetime.now().strftime('%B_%d_%Y_%H_%M_%S')}"

    return prefix


def get_description(area, initial_params, learning_rates, target_weights, train_size, val_len, der_1st_reg,
                    t_inc, momentum, m, a, loss_type, integrator, bound_reg, bound_loss_type,
                    model_cls
                    ):
    return {
        "started": datetime.now().strftime('%d/%B/%Y %H:%M:%S'),
        "model_cls": model_cls,
        "region": area,
        "learning_rates": learning_rates,
        "target_weights": target_weights,
        "train_size": train_size,
        "val_len": val_len,
        "der_1st_reg": der_1st_reg,
        "integrator": integrator.__name__,
        "t_inc": t_inc,
        "momentum": momentum,
        "m": m if momentum else None,
        "a": a if momentum else None,
        "loss_type": loss_type,
        "bound_reg": bound_reg,
        "bound_loss_type": bound_loss_type,
        "initial_values": initial_params
    }

def get_markdown_description(description_json, exp_id):
    markdown = f"## Experiment {exp_id}\n\n"
    description_md = description_json.replace("\n", "<br>").replace(" ", "&nbsp;")
    description_md = description_md.replace("[", "\\[").replace("]", "\\]")

    return markdown + description_md


def get_tabs(tabIdx):
    return '&emsp;' * tabIdx


def get_html_str_from_dict(dictionary, tabIdx=1):
    dict_str = "{<br>"
    for key, value in dictionary.items():
        dict_str += f"{get_tabs(tabIdx)}{key}:"
        if not isinstance(value, dict):
            dict_str += f"{value},<br>"
        else:
            dict_str += get_html_str_from_dict(value, tabIdx + 1) + ",<br>"
    dict_str += get_tabs(tabIdx - 1) + "}"
    return dict_str


def get_exp_description_html(description, uuid):
    """
    creates an html representation of the experiment description for tensorboard
    """

    description_str = f"Experiment id: {uuid}<br><br>"
    description_str += get_html_str_from_dict(description)

    return description_str