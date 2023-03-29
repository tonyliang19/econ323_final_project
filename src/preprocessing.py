# Preprocessing and Feature engineering
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
# common preprocessor for data
def preprocessing(df, drop="RAD"):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove(drop)
    
    # transformers
    numeric_transformer = make_pipeline(SimpleImputer(strategy="median"),
                                        StandardScaler())
    
    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_cols),# scaling on numeric features
        ("drop", [drop]) # drop RAD, since it is index-like obj
    )
    return preprocessor