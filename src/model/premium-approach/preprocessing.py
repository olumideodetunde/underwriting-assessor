# Imports
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin    
from sklearn.impute import SimpleImputer
from feature_eng import feature_engineering
from datasets import insurance_new as dataset
from sklearn.model_selection import train_test_split
import category_encoders as ce

# load the dataset
feature, target = feature_engineering(dataset)

# split the dataset into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(
    feature, target, test_size=0.1, random_state=42, shuffle=False
)

# Preprocessor class for data preprocessing
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_smoothing=0.3,
        min_samples_leaf=20,
        handle_unknown="value"
    ):
        # Store parameters EXACTLY as passed
        self.target_smoothing = target_smoothing
        self.min_samples_leaf = min_samples_leaf
        self.handle_unknown = handle_unknown

        # Define components
        self.num_imputer = SimpleImputer(strategy="mean")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = ce.TargetEncoder(
            smoothing=self.target_smoothing,
            min_samples_leaf=self.min_samples_leaf,
            handle_unknown=self.handle_unknown
        )


    def fit(self, X, y):
        if y is None:
            raise ValueError("Target y must be provided for target encoding.")

        # Lock schema
        self.num_cols_ = (
            X.select_dtypes(include=["int64", "float64"])
            .columns
            .tolist()
        )
        self.cat_cols_ = (
            X.select_dtypes(include=["object", "category"])
            .columns
            .tolist()
        )

        self.feature_names_ = self.num_cols_ + self.cat_cols_

        # Fit imputers
        self.num_imputer.fit(X[self.num_cols_])
        self.cat_imputer.fit(X[self.cat_cols_])

        # Fit target encoder on imputed categorical data
        X_cat_imputed = pd.DataFrame(
            self.cat_imputer.transform(X[self.cat_cols_]),
            columns=self.cat_cols_,
            index=X.index
        )

        self.encoder.fit(X_cat_imputed, y)

        return self

    def transform(self, X):
        # Schema validation
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns at transform time: {missing}")

        # Enforce column order and drop extras
        X = X[self.feature_names_]

        # Impute
        X_num = pd.DataFrame(
            self.num_imputer.transform(X[self.num_cols_]),
            columns=self.num_cols_,
            index=X.index
        )

        X_cat = pd.DataFrame(
            self.cat_imputer.transform(X[self.cat_cols_]),
            columns=self.cat_cols_,
            index=X.index
        )

        # Encode
        X_cat_enc = pd.DataFrame(
            self.encoder.transform(X_cat),
            columns=self.cat_cols_,
            index=X.index
        )

        return pd.concat([X_num, X_cat_enc], axis=1)
    
    # Get feature names after transformation
    def get_feature_names_out(self):
        return self.num_cols_ + self.cat_cols_
    
if __name__ == "__main__":
    print("Preprocessing module loaded successfully.")
    
    