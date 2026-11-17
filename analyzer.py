"""
analyzer.py
-----------
Module 3: Exploratory Data Analysis & Statistical Summaries

This module defines the DataAnalyzer class, responsible for:
- Summary statistics (mean, median, std, quartiles)
- Group-wise comparisons (defaulters vs non-defaulters)
- Correlation analysis
- Distribution inspection

Professional practice: every analysis method returns a plain Python
object (DataFrame, dict, or scalar) so that results can be used in
either the report or the visualizer without tight coupling.
"""

import pandas as pd
import numpy as np


class DataAnalyzer:
    """
    Performs statistical analysis on the cleaned credit dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data produced by DataPreprocessor.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Ensure DEFAULT is numeric for group-by math
        self.df["DEFAULT_NUM"] = self.df["DEFAULT"].astype(int)

    # ------------------------------------------------------------------
    # 3.1  Summary statistics
    # ------------------------------------------------------------------

    def numerical_summary(self, cols: list = None) -> pd.DataFrame:
        """
        Return descriptive statistics for numerical columns.

        Parameters
        ----------
        cols : list, optional
            Subset of columns. Defaults to all numeric.

        Returns
        -------
        pd.DataFrame  Transposed describe() with added median column.
        """
        if cols is None:
            cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in cols if c != "DEFAULT_NUM"]

        desc = self.df[cols].describe().T
        desc["median"] = self.df[cols].median()
        desc["skewness"] = self.df[cols].skew()
        return desc.round(2)

    def categorical_summary(self, cols: list = None) -> dict:
        """
        Return value-count tables for categorical columns.

        Returns
        -------
        dict  {column_name: pd.DataFrame with count and pct columns}
        """
        if cols is None:
            cols = self.df.select_dtypes(include=["category", "object"]).columns.tolist()
            cols = [c for c in cols if c != "DEFAULT"]

        result = {}
        for col in cols:
            counts = self.df[col].value_counts(dropna=False)
            pct    = (counts / len(self.df) * 100).round(2)
            result[col] = pd.DataFrame({"count": counts, "pct": pct})
        return result

    # ------------------------------------------------------------------
    # 3.2  Group-wise analysis: defaulters vs non-defaulters
    # ------------------------------------------------------------------

    def default_rate_by_group(self, groupby_col: str) -> pd.DataFrame:
        """
        Compute default rate (%) within each category of groupby_col.

        Parameters
        ----------
        groupby_col : str
            A categorical column (e.g. 'EDUCATION', 'SEX', 'AGE_GROUP').

        Returns
        -------
        pd.DataFrame  with columns [groupby_col, count, default_rate_pct]
        """
        grp = self.df.groupby(groupby_col, observed=True)
        agg = grp["DEFAULT_NUM"].agg(count="count", default_rate="mean").reset_index()
        agg["default_rate_pct"] = (agg["default_rate"] * 100).round(2)
        agg.drop(columns="default_rate", inplace=True)
        return agg.sort_values("default_rate_pct", ascending=False)

    def group_means(self, numerical_cols: list = None) -> pd.DataFrame:
        """
        Compare means of numerical columns between defaulters/non-defaulters.

        Returns
        -------
        pd.DataFrame  Columns: feature, mean_no_default, mean_default, diff_pct
        """
        if numerical_cols is None:
            numerical_cols = [
                "LIMIT_BAL", "AGE", "AVG_BILL_AMT", "AVG_PAY_AMT",
                "UTIL_RATE", "PAY_RATIO", "CONSEC_DELAYS",
            ]
        numerical_cols = [c for c in numerical_cols if c in self.df.columns]

        rows = []
        for col in numerical_cols:
            m0 = self.df.loc[self.df["DEFAULT_NUM"] == 0, col].mean()
            m1 = self.df.loc[self.df["DEFAULT_NUM"] == 1, col].mean()
            diff = ((m1 - m0) / (abs(m0) + 1e-9)) * 100
            rows.append({
                "feature":          col,
                "mean_no_default":  round(m0, 2),
                "mean_default":     round(m1, 2),
                "diff_pct":         round(diff, 2),
            })
        return pd.DataFrame(rows).sort_values("diff_pct", key=abs, ascending=False)

    # ------------------------------------------------------------------
    # 3.3  Correlation analysis
    # ------------------------------------------------------------------

    def correlation_matrix(self, cols: list = None) -> pd.DataFrame:
        """
        Pearson correlation matrix for selected numerical columns.

        Parameters
        ----------
        cols : list, optional
            Columns to include.

        Returns
        -------
        pd.DataFrame  Symmetric correlation matrix.
        """
        if cols is None:
            cols = [
                "LIMIT_BAL", "AGE", "AVG_BILL_AMT", "AVG_PAY_AMT",
                "UTIL_RATE", "PAY_RATIO", "CONSEC_DELAYS", "DEFAULT_NUM",
            ]
        cols = [c for c in cols if c in self.df.columns]
        return self.df[cols].corr().round(3)

    def top_correlations_with_default(self, n: int = 10) -> pd.DataFrame:
        """
        Return the n features most correlated with DEFAULT (absolute value).

        Returns
        -------
        pd.DataFrame  Columns: feature, correlation
        """
        corr = self.correlation_matrix()
        if "DEFAULT_NUM" not in corr.columns:
            return pd.DataFrame()
        series = corr["DEFAULT_NUM"].drop("DEFAULT_NUM").abs().sort_values(ascending=False)
        return pd.DataFrame({
            "feature":     series.index,
            "abs_corr_with_default": series.values.round(4),
        }).head(n)

    # ------------------------------------------------------------------
    # 3.4  Repayment behaviour analysis
    # ------------------------------------------------------------------

    def repayment_status_distribution(self) -> pd.DataFrame:
        """
        Compute the percentage distribution of each PAY status code
        across all six months.

        Returns
        -------
        pd.DataFrame  Multi-indexed by (month, status_code).
        """
        pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        pay_cols = [c for c in pay_cols if c in self.df.columns]
        rows = []
        for col in pay_cols:
            vc = self.df[col].value_counts(normalize=True).mul(100).round(2)
            for code, pct in vc.items():
                rows.append({"month": col, "status_code": code, "pct": pct})
        return pd.DataFrame(rows)

    def default_rate_by_delay_count(self) -> pd.DataFrame:
        """
        Default rate grouped by CONSEC_DELAYS (0 to 6 months of delay).

        Returns
        -------
        pd.DataFrame
        """
        if "CONSEC_DELAYS" not in self.df.columns:
            raise ValueError("Run DataPreprocessor.run_pipeline() first.")
        grp = self.df.groupby("CONSEC_DELAYS")["DEFAULT_NUM"]
        agg = grp.agg(count="count", default_rate="mean").reset_index()
        agg["default_rate_pct"] = (agg["default_rate"] * 100).round(2)
        return agg

    # ------------------------------------------------------------------
    # Convenience: print all key findings to stdout
    # ------------------------------------------------------------------

    def print_key_findings(self) -> None:
        print("\n" + "="*60)
        print("  KEY ANALYTICAL FINDINGS")
        print("="*60)

        overall_rate = self.df["DEFAULT_NUM"].mean() * 100
        print(f"\nOverall default rate : {overall_rate:.2f}%")

        print("\n-- Top Features Correlated with Default --")
        print(self.top_correlations_with_default(8).to_string(index=False))

        print("\n-- Mean Comparison: Defaulters vs Non-Defaulters --")
        print(self.group_means().to_string(index=False))

        print("\n-- Default Rate by Education --")
        print(self.default_rate_by_group("EDUCATION").to_string(index=False))

        print("\n-- Default Rate by Consecutive Delays --")
        print(self.default_rate_by_delay_count().to_string(index=False))
        print("="*60 + "\n")
