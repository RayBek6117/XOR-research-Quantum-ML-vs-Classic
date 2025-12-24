def accuracy(circuit, params, X_angles, y, threshold: float = 0.5):

    """
    Compute classification accuracy using p(class=1) >= threshold.

    Returns
    -------
    acc : float
    """

    correct = 0
    for i in range(len(X_angles)):
        expval = circuit(params, X_angles[i])
        p1 = float((1.0 - expval) / 2.0)
        pred = 1 if p1 >= threshold else 0
        if pred == int(y[i]):
            correct += 1
    return correct / len(X_angles)
