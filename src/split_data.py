import pandas as pd

# function to load and split data

# loads data from path, and specifies proportion of train data, with 1 - proportion of test data
# and target is the variable of interest (your y), then returns train and test data
# proportion is default to 0.5, and target default to None 
def split_data(data_path, proportion=0.5, target=None, random_state=123):
    """
    Loads data from path, and specifies proportion of train data, with 1 - proportion of test data
    and target is the variable of interest (your y), then returns train and test data
    proportion is default to 0.5, and target default to None.
    
    Optional argument:
    random_state = 123 (default), change to other number of your choice to assert reproducibility
    """
    # load the data
    data = pd.read_csv(data_path)
    # drop nas
    data = data.dropna()
    # inner function to split data into train and test portion
    def train_test_split(data, proportion):
        train = data.sample(frac = proportion, random_state=random_state)
        test = data.drop(train.index)
        # rest and remove index of both
        train = train.reset_index().drop(columns=["index"])
        test = test.reset_index().drop(columns=["index"])
        # asserting dimension matches (i.e. number of rows)
        assert train.shape[0] + test.shape[0] == data.shape[0]
        return train, test
    # split the data into train and test
    train, test = train_test_split(data, proportion)
    # further split train data to X and y
    def split_X_y(data):
        X = data.drop(columns=[target])
        y = data[target]
        return X, y
    X_train, y_train = split_X_y(train)
    # split test data to X and y
    X_test, y_test = split_X_y(test)
    # check dimension again
    assert X_train.shape[0] + X_test.shape[0] == data.shape[0]
    assert X_train.shape[1] and X_test.shape[1] == 13
    assert y_train.shape[0] == X_train.shape[0] and y_test.shape[0] == X_test.shape[0]
    # return the objects needed
    return X_train, X_test, y_train, y_test