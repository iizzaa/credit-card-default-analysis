"""
data_loader.py
--------------
Module 1: Data Loading and Understanding

This module defines the CreditDataLoader class, responsible for:
- Loading the UCI Credit Card Default dataset
- Providing a structured summary of the data
- Identifying data quality challenges (missing values, types, etc.)

Professional practice: encapsulating all I/O operations in one class
makes it easy to swap the data source without touching other modules.
"""

import pandas as pd
import numpy as np


class CreditDataLoader:
    """
    Handles loading and initial inspection of the credit card default dataset.

    Attributes
    ----------
    filepath : str
        Path to the CSV file.
    df : pd.DataFrame or None
        The loaded dataset (None until load() is called).
    """

    # Column metadata: human-readable names and expected types
    COLUMN_METADATA = {
        "ID":             ("Client ID",                          "identifier"),
        "LIMIT_BAL":      ("Credit Limit (NT$)",                 "numerical"),
        "SEX":            ("Sex (1=Male, 2=Female)",             "categorical"),
        "EDUCATION":      ("Education Level",                    "categorical"),
        "MARRIAGE":       ("Marital Status",                     "categorical"),
        "AGE":            ("Age (years)",                        "numerical"),
        "PAY_0":          ("Repayment Status Sep",               "categorical"),
        "PAY_2":          ("Repayment Status Aug",               "categorical"),
        "PAY_3":          ("Repayment Status Jul",               "categorical"),
        "PAY_4":          ("Repayment Status Jun",               "categorical"),
        "PAY_5":          ("Repayment Status May",               "categorical"),
        "PAY_6":          ("Repayment Status Apr",               "categorical"),
        "BILL_AMT1":      ("Bill Amount Sep (NT$)",              "numerical"),
        "BILL_AMT2":      ("Bill Amount Aug (NT$)",              "numerical"),
        "BILL_AMT3":      ("Bill Amount Jul (NT$)",              "numerical"),
        "BILL_AMT4":      ("Bill Amount Jun (NT$)",              "numerical"),
        "BILL_AMT5":      ("Bill Amount May (NT$)",              "numerical"),
        "BILL_AMT6":      ("Bill Amount Apr (NT$)",              "numerical"),
        "PAY_AMT1":       ("Payment Amount Sep (NT$)",           "numerical"),
        "PAY_AMT2":       ("Payment Amount Aug (NT$)",           "numerical"),
        "PAY_AMT3":       ("Payment Amount Jul (NT$)",           "numerical"),
        "PAY_AMT4":       ("Payment Amount Jun (NT$)",           "numerical"),
        "PAY_AMT5":       ("Payment Amount May (NT$)",           "numerical"),
        "PAY_AMT6":       ("Payment Amount Apr (NT$)",           "numerical"),
        "default.payment.next.month": ("Default Next Month (0=No, 1=Yes)", "target"),
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load the dataset from CSV.  The UCI file has a header row on row 0
        and a second descriptive row on row 1 that must be skipped.

        Returns
        -------
        pd.DataFrame
            The raw, unmodified dataset.
        """
        self.df = pd.read_csv(self.filepath)
        # Rename target column for convenience
        self.df.rename(columns={"default.payment.next.month": "DEFAULT"}, inplace=True)
        print(f"[DataLoader] Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
        return self.df

    def shape_summary(self) -> dict:
        """Return a dict with basic shape info."""
        self._require_loaded()
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1e6, 2),
        }

    def missing_values_report(self) -> pd.DataFrame:
        """
        Compute missing-value counts and percentages for every column.

        Returns
        -------
        pd.DataFrame
            Sorted by missing count descending.
        """
        self._require_loaded()
        missing = self.df.isnull().sum()
        pct     = (missing / len(self.df) * 100).round(2)
        report  = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        return report.sort_values("missing_count", ascending=False)

    def dtype_report(self) -> pd.DataFrame:
        """
        Report inferred dtypes alongside expected semantic types.

        Returns
        -------
        pd.DataFrame
        """
        self._require_loaded()
        rows = []
        for col in self.df.columns:
            label, sem_type = self.COLUMN_METADATA.get(col, (col, "unknown"))
            rows.append({
                "column":        col,
                "label":         label,
                "semantic_type": sem_type,
                "pandas_dtype":  str(self.df[col].dtype),
                "unique_values": self.df[col].nunique(),
            })
        return pd.DataFrame(rows)

    def outlier_report(self, numerical_cols: list = None) -> pd.DataFrame:
        """
        Detect outliers in numerical columns using the IQR method.

        Parameters
        ----------
        numerical_cols : list, optional
            Columns to check. Defaults to all float64/int64 columns
            except ID and DEFAULT.

        Returns
        -------
        pd.DataFrame
            Per-column outlier counts.
        """
        self._require_loaded()
        if numerical_cols is None:
            numerical_cols = [
                c for c in self.df.select_dtypes(include=[np.number]).columns
                if c not in ("ID", "DEFAULT")
            ]

        rows = []
        for col in numerical_cols:
            q1, q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            iqr    = q3 - q1
            lower  = q1 - 1.5 * iqr
            upper  = q3 + 1.5 * iqr
            n_out  = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            rows.append({
                "column":      col,
                "q1":          round(q1, 2),
                "q3":          round(q3, 2),
                "lower_fence": round(lower, 2),
                "upper_fence": round(upper, 2),
                "outlier_count": int(n_out),
                "outlier_pct":   round(n_out / len(self.df) * 100, 2),
            })
        return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)

    def categorical_anomalies(self) -> dict:
        """
        Check categorical columns for undocumented category codes.

        Valid codes per the dataset description:
          SEX       : {1, 2}
          EDUCATION : {1, 2, 3, 4}   (0, 5, 6 are undocumented)
          MARRIAGE  : {1, 2, 3}      (0 is undocumented)
          PAY_*     : {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

        Returns
        -------
        dict  column -> set of invalid values found
        """
        self._require_loaded()
        valid_codes = {
            "SEX":       {1, 2},
            "EDUCATION": {1, 2, 3, 4},
            "MARRIAGE":  {1, 2, 3},
        }
        anomalies = {}
        for col, valid in valid_codes.items():
            if col in self.df.columns:
                found = set(self.df[col].unique()) - valid
                if found:
                    anomalies[col] = found
        return anomalies

    def full_report(self) -> None:
        """Print a human-readable summary of all data-quality findings."""
        self._require_loaded()

        print("\n" + "="*60)
        print("  DATASET OVERVIEW")
        print("="*60)
        s = self.shape_summary()
        print(f"  Rows      : {s['rows']:,}")
        print(f"  Columns   : {s['columns']}")
        print(f"  Memory    : {s['memory_mb']} MB")

        print("\n--- Missing Values ---")
        mv = self.missing_values_report()
        non_zero = mv[mv["missing_count"] > 0]
        if non_zero.empty:
            print("  No missing values found.")
        else:
            print(non_zero.to_string())

        print("\n--- Data Types ---")
        print(self.dtype_report().to_string(index=False))

        print("\n--- Categorical Anomalies ---")
        anom = self.categorical_anomalies()
        if not anom:
            print("  No undocumented category codes found.")
        else:
            for col, vals in anom.items():
                print(f"  {col}: undocumented codes → {vals}")

        print("\n--- Outlier Summary (top 5 columns) ---")
        print(self.outlier_report().head(10).to_string(index=False))
        print("="*60 + "\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_loaded(self):
        if self.df is None:
            raise RuntimeError("Call load() before accessing data.")
