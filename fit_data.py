import os
from torch.optim import Adam, SGD
import skopt
import torch

from utils.data_utils import select_data
from utils.visualization_utils import plot_data_and_fit
from learning_models.logistic import Logistic


df_file = os.path.join(os.getcwd(), "dati-regioni", "dpc-covid19-ita-regioni.csv")
area = ["Sardegna"]  # list(df["denominazione_regione"].unique())
area_col_name = "denominazione_regione"
value_col_name = "deceduti"

configs = {"optimizer": SGD, "n_epochs": 10000}

x, y = select_data(df_file, area, area_col_name, value_col_name, file_sep=",")
LOGISTIC_MODEL = Logistic((x, y), configs)

def train(params):
    return LOGISTIC_MODEL.fit(params)


SPACE = [skopt.space.Real(1e-9, 1e-5, name='lrw', prior='log-uniform'),
         skopt.space.Real(1e-9, 1e-5, name='lrb', prior='log-uniform'),
         skopt.space.Real(1e-3, 1e-1, name='lrm', prior='log-uniform'),
         skopt.space.Real(-1.0, 1.0, name='initial_w', prior='uniform'),
         skopt.space.Real(-1.0, 1.0, name='initial_b', prior='uniform'),
         skopt.space.Real(min(y)/10, max(y), name='initial_m', prior='uniform'),
         ]


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    all_params = {**params}
    return train(all_params)


res_gp = skopt.gp_minimize(objective, SPACE, n_calls=30)  # n_calls is the number of repeated trials
# print(res_gp)
score = "Best score=%.4f" % res_gp.fun
result = """Best parameters:
- lrw=%.9f
- lrb=%.9f
- lrm=%.9f
- initial_w=%.6f
- initial_b=%.6f
- initial_m=%.6f""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2], res_gp.x[3], res_gp.x[4], res_gp.x[5])

print(score)
print(result)
#
base_path = os.path.join(os.getcwd(), "regioni")
if not os.path.exists(base_path):
    os.mkdir(base_path)

log_file = os.path.join(base_path, area[0] + "_best_results.txt")
with open(log_file, "w") as f:
    f.write(score)
    f.write(result)

y_hat = LOGISTIC_MODEL(LOGISTIC_MODEL.x).detach().numpy()
data = (LOGISTIC_MODEL.x.detach().numpy(), LOGISTIC_MODEL.y.detach().numpy())

future_days = 30  # predictions for the future 30 days and current date
future_x = torch.tensor([i+len(y) for i in range(future_days)]).view(-1, 1).float()
future_y = LOGISTIC_MODEL(future_x).detach().numpy()
future_x = future_x.detach().numpy()

save_plot_path = os.path.join(base_path, area[0] + ".png")
plot_data_and_fit(data, fitted_data=(x, y_hat), future_data=(future_x, future_y), save_path=save_plot_path, plot_name=area[0])
