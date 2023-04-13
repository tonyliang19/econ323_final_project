import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
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


# helper function to calculate mean and stf for cv scores
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []
    for i in range(len(mean_scores)):
        if i > 1 and i < 8:
            out_col.append((f"%0.3f (+/- %0.3f)" % (-1 * mean_scores[i], std_scores[i])))
        else:
            out_col.append((f"%0.3f (+/- %0.3f)" % (1 * mean_scores[i], std_scores[i])))
    return pd.Series(data=out_col, index=mean_scores.index)