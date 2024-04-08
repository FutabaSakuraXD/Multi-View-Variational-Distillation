import numpy as np

from reid.models.baseline import Baseline
from reid.models.newresnet import VIB, MIEstimator


############################################################################################
# Parameters count
############################################################################################
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
############################################################################################

file_name_1 = "/home/txd/Variational Distillation/Exps/FSE_6/log"
time_cost_1 = []
with open(file_name_1, 'r') as f:
    count = 0
    for line in f:
        count += 1
        if line.find("Epoch") != 0: continue
        if count >= 300: break
        print(line[line.find("Time") + 11: line.find("Time") + 15])
        time_cost_1.append(24 * float(line[line.find("Time") + 11: line.find("Time") + 15]))
time_cost = np.asarray(time_cost_1)
sum_1 = time_cost.sum()

file_name_2 = "/home/txd/Variational Distillation/Exps/FSE_5/log"
time_cost_2 = []
with open(file_name_2, 'r') as f:
    count = 0
    for line in f:
        count += 1
        if line.find("Epoch") != 0: continue
        if count >= 300: break
        print(line[line.find("Time") + 11: line.find("Time") + 15])
        time_cost_2.append(24 * float(line[line.find("Time") + 11: line.find("Time") + 15]))
time_cost = np.asarray(time_cost_2)
sum_2 = time_cost.sum()

baseline = Baseline(num_classes=395, num_features=2048)
vib = VIB()
mie = MIEstimator()
print("Params of an encoder: " + str(count_param(baseline)))
print("Params of an information bottleneck: " + str(count_param(vib)))
print("Params of an mutual information estimator: " + str(count_param(mie)))

# Params:
# w/o estimator: 88241631
# with estimator: 107140578