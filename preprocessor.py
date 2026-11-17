"""
preprocessor.py
---------------
Module 2: Data Preprocessing

This module defines the DataPreprocessor class, responsible for:
- Handling undocumented / invalid categorical codes
- Capping outliers using the IQR method
- Feature engineering (derived columns)
- Data-type corrections
- Producing a clean, analysis-ready DataFrame

Professional practice: all transformations are implemented as small,
single-purpose methods that are chained inside a public `run_pipeline()`
method.  This makes it trivial to enable/disable individual steps and
unit-test each transformation in isolation.
"""

import pandas as pd
import numpy as np


class DataPreprocessor:
    """
    Cleans and transforms the raw credit card default DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by CreditDataLoader.

    Attributes
    ----------
    df_clean : pd.DataFrame
        The cleaned/transformed dataset after run_pipeline() completes.
    log : list[str]
        Audit trail — one entry per transformation step.
    """

    # Mapping undocumented codes → documented "other" bucket
    EDUCATION_REMAP = {0: 4, 5: 4, 6: 4}   # 4 = "other"
    MARRIAGE_REMAP  = {0: 3}                 # 3 = "other"

    # Human-readable labels for reporting / plotting
    SEX_LABELS       = {1: "Male",   2: "Female"}
    EDUCATION_LABELS = {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
    MARRIAGE_LABELS  = {1: "Married",  2: "Single",     3: "Other"}
    DEFAULT_LABELS   = {0: "No Default", 1: "Default"}

    PAY_COLS  = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    def __init__(self, df: pd.DataFrame):
        self.df_clean = df.copy()
        self.log: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute all preprocessing steps in order.

        Returns
        -------
        pd.DataFrame  The cleaned dataset.
        """
        self._drop_id_column()
        self._fix_categorical_codes()
        self._convert_categorical_dtypes()
        self._cap_outliers_iqr()
        self._engineer_features()
        self._log("Pipeline complete.")
        self._print_log()
        return self.df_clean

    # ------------------------------------------------------------------
    # Step implementations (private — called only by run_pipeline)
    # ------------------------------------------------------------------

    def _drop_id_column(self):
        """Remove the ID column — it carries no predictive information."""
        if "ID" in self.df_clean.columns:
            self.df_clean.drop(columns=["ID"], inplace=True)
            self._log("Dropped ID column (identifier, not a feature).")

    def _fix_categorical_codes(self):
        """
        Remap undocumented category codes to the nearest valid bucket.

        EDUCATION: codes 0, 5, 6 are not in the dataset description.
                   We remap them to 4 ('other') to preserve the rows.
        MARRIAGE : code 0 is undocumented → remapped to 3 ('other').
        """
        before_edu = self.df_clean["EDUCATION"].value_counts().get(0, 0)
        self.df_clean["EDUCATION"] = self.df_clean["EDUCATION"].replace(self.EDUCATION_REMAP)
        self._log(
            f"EDUCATION: remapped codes {{0,5,6}} → 4 (other). "
            f"Affected rows: {before_edu}."
        )

        before_mar = self.df_clean["MARRIAGE"].value_counts().get(0, 0)
        self.df_clean["MARRIAGE"] = self.df_clean["MARRIAGE"].replace(self.MARRIAGE_REMAP)
        self._log(
            f"MARRIAGE: remapped code {{0}} → 3 (other). "
            f"Affected rows: {before_mar}."
        )

    def _convert_categorical_dtypes(self):
        """
        Convert integer-coded categoricals to pandas Categorical dtype.
        This reduces memory and makes intent explicit to other developers.
        """
        cat_cols = {"SEX": None, "EDUCATION": None, "MARRIAGE": None, "DEFAULT": None}
        for col in cat_cols:
            if col in self.df_clean.columns:
                self.df_clean[col] = self.df_clean[col].astype("category")
        self._log("Converted SEX, EDUCATION, MARRIAGE, DEFAULT → Categorical dtype.")

    def _cap_outliers_iqr(self, factor: float = 3.0):
        """
        Cap (winsorise) extreme outliers in numeric bill and payment columns.

        We use a factor of 3.0 (instead of the standard 1.5) because credit
        card amounts are naturally right-skewed; a tighter fence would
        discard legitimate high-value transactions.

        Parameters
        ----------
        factor : float
            IQR multiplier for the fence.
        """
        target_cols = self.BILL_COLS + self.PAY_AMT_COLS + ["LIMIT_BAL", "AGE"]
        n_capped = 0
        for col in target_cols:
            if col not in self.df_clean.columns:
                continue
            q1 = self.df_clean[col].quantile(0.25)
            q3 = self.df_clean[col].quantile(0.75)
            iqr   = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            mask  = (self.df_clean[col] < lower) | (self.df_clean[col] > upper)
            n_capped += mask.sum()
            self.df_clean[col] = self.df_clean[col].clip(lower, upper)
        self._log(f"Outlier capping (IQR × {factor}): {n_capped:,} values clipped across numeric columns.")

    def _engineer_features(self):
        """
        Create derived features that may improve analytical insight.

        New columns:
          AVG_BILL_AMT   : Mean bill amount over 6 months.
          AVG_PAY_AMT    : Mean payment amount over 6 months.
          UTIL_RATE      : Average utilisation = AVG_BILL_AMT / LIMIT_BAL.
          PAY_RATIO      : AVG_PAY_AMT / (AVG_BILL_AMT + 1) to avoid ÷0.
          CONSEC_DELAYS  : Number of months with payment delay ≥ 1.
          AGE_GROUP      : Binned age into four life-stage buckets.
        """
        df = self.df_clean

        df["AVG_BILL_AMT"] = df[self.BILL_COLS].mean(axis=1).round(2)
        df["AVG_PAY_AMT"]  = df[self.PAY_AMT_COLS].mean(axis=1).round(2)

        df["UTIL_RATE"] = (df["AVG_BILL_AMT"] / df["LIMIT_BAL"].replace(0, np.nan)).round(4)
        df["UTIL_RATE"] = df["UTIL_RATE"].fillna(0)

        df["PAY_RATIO"] = (df["AVG_PAY_AMT"] / (df["AVG_BILL_AMT"] + 1)).round(4)

        df["CONSEC_DELAYS"] = (df[self.PAY_COLS] >= 1).sum(axis=1)

        df["AGE_GROUP"] = pd.cut(
            df["AGE"],
            bins=[0, 25, 35, 50, 100],
            labels=["≤25", "26-35", "36-50", "51+"],
        )

        self._log(
            "Engineered 6 new features: AVG_BILL_AMT, AVG_PAY_AMT, "
            "UTIL_RATE, PAY_RATIO, CONSEC_DELAYS, AGE_GROUP."
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log.append(msg)

    def _print_log(self):
        print("\n" + "="*60)
        print("  PREPROCESSING LOG")
        print("="*60)
        for i, entry in enumerate(self.log, 1):
            print(f"  Step {i:02d}: {entry}")
        print("="*60 + "\n")

    def get_numerical_cols(self) -> list:
        """Return names of all remaining numerical columns (post-preprocessing)."""
        return self.df_clean.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_cols(self) -> list:
        """Return names of all categorical columns (post-preprocessing)."""
        return self.df_clean.select_dtypes(include=["category"]).columns.tolist()

    def summary(self) -> None:
        """Print shape and dtype breakdown after preprocessing."""
        print(f"Clean dataset: {self.df_clean.shape[0]:,} rows × {self.df_clean.shape[1]} columns")
        print(f"Numerical columns   : {len(self.get_numerical_cols())}")
        print(f"Categorical columns : {len(self.get_categorical_cols())}")
        print(f"Default rate        : {(self.df_clean['DEFAULT'] == 1).mean()*100:.2f}%")
