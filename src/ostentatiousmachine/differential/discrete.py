def diff(t, x):
    # Check that the time and signal arrays have equal lengths
    if len(t) != len(x):
        raise ValueError("t and x must have equal lengths")

    # A derivative requires at least two data points
    if len(t) < 2:
        return []

    v = []

    # Compute the discrete derivative between consecutive points
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]

        if dt == 0:
            raise ValueError("Consecutive time values cannot be equal")

        derivative = (x[k] - x[k - 1]) / dt
        v.append(derivative)

    return v
