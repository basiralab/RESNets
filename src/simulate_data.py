import numpy as np

def simulate_data(seed = 123):
    n_s = int(input("Size of baseline population:" ))
    n_t = int(input("Number of following timepoints: "))
    n_r = int(input("Number of nodes in the graph(e.g. ROIs): "))
    feature_vec_length = int(n_r * (n_r - 1) / 2)

    np.random.seed(seed)
    mean_test = np.random.rand()
    std_test = np.random.rand()
    test_subject = np.abs(np.random.normal(mean_test, std_test, (1, feature_vec_length)))

    mean_baseline = np.random.rand()
    std_baseline = np.random.rand()
    baseline_population = np.abs(np.random.normal(mean_baseline, std_baseline, (n_s, feature_vec_length)))

    means_timepoints = np.random.rand(n_t)
    stds_timepoints = np.random.rand(n_t)
    follow_up_data = np.zeros((n_t, n_s, feature_vec_length))
    i = 0
    for m, s in zip(means_timepoints, stds_timepoints):
        follow_up_data[i] = np.abs(np.random.normal(m, s, (n_s, feature_vec_length)))
        i += 1


    return test_subject, baseline_population, follow_up_data, n_r