import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

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

# helper to fit a model based on name and optional keyword arguments provided to models
def fit_raw_model(X_train, y_train, X_test, y_test, name="", **kwargs):
    # check valid model names
    valid = ["OLS", "LASSO", "RF"]
    if name not in valid:
        print("Did not provide a correct model name, try again")
        return None
    # instantiate regressor by name provide
    if name == "OLS":
        mod = LinearRegression()
    if name == "LASSO":
        mod = Lasso(**kwargs)
    if name == "RF":
        name = "Random Forest"
        mod = RandomForestRegressor(**kwargs)
    
    # fit to train data and predict on test
    mod.fit(X_train, y_train)
    predicted = mod.predict(X_test)
    # call helper to get results on scoring metrics on test set
    result = get_metrics(actual=y_test, predicted=predicted, name=name)
    return result.style.set_caption(f"Test Scores on {name} Regression")

# helper to get metrics for models
def get_metrics(actual, predicted, name=""):
    # calculate MSE
    MSE = np.square(np.subtract(actual, predicted)).mean() 
    # calculate RMSE
    RMSE = np.sqrt(MSE)
    # calculate MAPE
    MAPE = np.mean(np.abs((actual - predicted)/actual))
    # store result
    out = {"RMSE": round(RMSE, 3), 
           "MSE": round(MSE, 3),
           "MAPE": round(MAPE,3)
          }
    # convert to dataframe with index name equal to name of model
    result = pd.DataFrame([out], index=[name])
    return result

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