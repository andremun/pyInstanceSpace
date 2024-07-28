def test_error_function():
    sd = SampleDataNum()

    mtr = MatlabResultsNum()

    X_sample = sd.X_sample
    Y_sample = sd.Y_sample
    n = X_sample.shape[1]
    m = X_sample.shape[1] + Y_sample.shape[1]  # Total number of features including appended Y

    # alpha_sample = mtr.data['X0']
    alpha_sample = mtr.data["alpha"][:, 0]
    x_bar_sample = np.hstack([X_sample, Y_sample])
    n = X_sample.shape[1]
    m = x_bar_sample.shape[1]

    pilot = Pilot()
    error = pilot.error_function(alpha_sample, x_bar_sample, n, m)


    matlab_error = mtr.data["eoptim"][0, 0]
    assert (error == matlab_error)
