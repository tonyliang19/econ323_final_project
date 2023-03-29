import numpy as np
# implementation of OLS in data
# known B = (XTX)^-1XT Y
def OLS(X_mat, y_mat):
    """
    Converts the parameters to numpy arrays and perform matrix multiplication to get betas of OLS from
    (X^TX)^-1 X^T y
    """
    # add intercept column to matrix X
    X = X_mat
    y = y_mat
    try:
        X.insert(0,'intercept',1)
    except:
        pass
    X = X.to_numpy()
    y = y.to_numpy()
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta[0], beta[1:]