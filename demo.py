from src.RESNets import RESNets
from src.simulate_data import simulate_data
from src.utilities import show_mtrx, to_2d
from matplotlib import pyplot as plt



test_subject, baseline_population, follow_up, n_r = simulate_data(seed = 123)

K = int(input("Number of selected subjects: "))
show_mtrx(to_2d(test_subject[0], n_r), "brain network of testing subject at baseline timepoint")

test_trajectory = RESNets(test_subject, baseline_population, follow_up, K, n_r)

t = 1
for pred in test_trajectory:
    show_mtrx(to_2d(pred[0], n_r), "testing subject at timepoint t_{}".format(t))
    t += 1
