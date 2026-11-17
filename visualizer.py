"""
visualizer.py
-------------
Module 4: Visualization

This module defines the Visualizer class, responsible for producing all
figures used in the report.  Each method corresponds to one figure.

Professional practice:
- A single `_save_fig()` helper enforces consistent saving conventions.
- Figure aesthetics (colors, font sizes) are set once in `__init__` via
  rcParams — so changing the style of every plot requires editing one place.
- Methods return the figure object so callers can further customise if needed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server/batch use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class Visualizer:
    """
    Produces and saves all project figures.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data from DataPreprocessor.
    output_dir : str
        Directory where PNG files will be saved.
    """

    # Consistent colour palette
    COLORS = {
        "no_default":  "#4C72B0",
        "default":     "#DD8452",
        "neutral":     "#55A868",
        "highlight":   "#C44E52",
        "grey":        "#8C8C8C",
    }

    def __init__(self, df: pd.DataFrame, output_dir: str = "outputs/figures"):
        self.df         = df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Global style settings applied to every figure
        plt.rcParams.update({
            "figure.dpi":        120,
            "font.size":         11,
            "axes.titlesize":    13,
            "axes.labelsize":    11,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "legend.frameon":    False,
        })
        # Numeric DEFAULT for plotting
        self.df["DEFAULT_NUM"] = self.df["DEFAULT"].astype(int)

    # ------------------------------------------------------------------
    # Figure 1  –  Default rate distribution (pie / bar)
    # ------------------------------------------------------------------

    def fig_default_distribution(self):
        """Bar chart showing count and rate of default vs non-default."""
        counts = self.df["DEFAULT_NUM"].value_counts().sort_index()
        labels = ["No Default", "Default"]
        colors = [self.COLORS["no_default"], self.COLORS["default"]]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")

        # Annotate bars with count and percentage
        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            pct = val / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10,
            )

        ax.set_title("Target Variable Distribution: Default vs No Default")
        ax.set_ylabel("Number of Clients")
        ax.set_ylim(0, counts.max() * 1.18)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        return self._save_fig(fig, "fig1_default_distribution.png")

    # ------------------------------------------------------------------
    # Figure 2  –  Credit limit distribution by default status
    # ------------------------------------------------------------------

    def fig_credit_limit_histogram(self):
        """Overlapping histograms of LIMIT_BAL for defaulters / non-defaulters."""
        no_def = self.df.loc[self.df["DEFAULT_NUM"] == 0, "LIMIT_BAL"]
        def_   = self.df.loc[self.df["DEFAULT_NUM"] == 1, "LIMIT_BAL"]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = np.linspace(0, self.df["LIMIT_BAL"].max(), 40)

        ax.hist(no_def, bins=bins, alpha=0.6, color=self.COLORS["no_default"],
                label="No Default", density=True)
        ax.hist(def_,   bins=bins, alpha=0.6, color=self.COLORS["default"],
                label="Default",    density=True)

        ax.set_title("Credit Limit Distribution by Default Status")
        ax.set_xlabel("Credit Limit (NT$)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
        fig.tight_layout()
        return self._save_fig(fig, "fig2_credit_limit_histogram.png")

    # ------------------------------------------------------------------
    # Figure 3  –  Default rate by education level
    # ------------------------------------------------------------------

    def fig_default_by_education(self):
        """Horizontal bar chart of default rate per education group."""
        edu_labels = {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
        df_tmp = self.df.copy()
        df_tmp["EDU_LABEL"] = df_tmp["EDUCATION"].astype(int).map(edu_labels)

        grp = (
            df_tmp.groupby("EDU_LABEL")["DEFAULT_NUM"]
            .mean()
            .mul(100)
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        colors  = [self.COLORS["default"] if v > 22 else self.COLORS["no_default"]
                   for v in grp.values]
        bars    = ax.barh(grp.index, grp.values, color=colors, edgecolor="white")

        for bar, val in zip(bars, grp.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)

        ax.set_title("Default Rate by Education Level")
        ax.set_xlabel("Default Rate (%)")
        ax.set_xlim(0, grp.max() * 1.2)
        fig.tight_layout()
        return self._save_fig(fig, "fig3_default_by_education.png")

    # ------------------------------------------------------------------
    # Figure 4  –  Default rate by age group
    # ------------------------------------------------------------------

    def fig_default_by_age_group(self):
        """Bar chart of default rate per age bucket."""
        if "AGE_GROUP" not in self.df.columns:
            print("[Visualizer] AGE_GROUP not found — skipping fig4.")
            return None

        grp = (
            self.df.groupby("AGE_GROUP", observed=True)["DEFAULT_NUM"]
            .mean().mul(100)
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(grp.index.astype(str), grp.values,
               color=self.COLORS["neutral"], edgecolor="white", width=0.5)

        for i, val in enumerate(grp.values):
            ax.text(i, val + 0.3, f"{val:.1f}%", ha="center", fontsize=10)

        ax.set_title("Default Rate by Age Group")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Default Rate (%)")
        ax.set_ylim(0, grp.max() * 1.2)
        fig.tight_layout()
        return self._save_fig(fig, "fig4_default_by_age_group.png")

    # ------------------------------------------------------------------
    # Figure 5  –  Correlation heat-map
    # ------------------------------------------------------------------

    def fig_correlation_heatmap(self, corr_matrix: pd.DataFrame):
        """
        Heatmap of the correlation matrix.

        Parameters
        ----------
        corr_matrix : pd.DataFrame
            Output of DataAnalyzer.correlation_matrix().
        """
        fig, ax = plt.subplots(figsize=(9, 7))
        data   = corr_matrix.values
        labels = corr_matrix.columns.tolist()
        n      = len(labels)

        # Manual colour mapping: red for positive, blue for negative
        cmap = plt.cm.RdBu_r
        im   = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)

        # Ticks
        ax.set_xticks(range(n));  ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(n));  ax.set_yticklabels(labels)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val   = data[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Pearson Correlation Matrix")
        fig.tight_layout()
        return self._save_fig(fig, "fig5_correlation_heatmap.png")

    # ------------------------------------------------------------------
    # Figure 6  –  Average bill amount over 6 months (line plot)
    # ------------------------------------------------------------------

    def fig_avg_bill_over_time(self):
        """
        Line plot of mean bill amount per month, split by default status.
        Shows temporal trend in spending behaviour.
        """
        bill_cols  = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                      "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
        month_lbls = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]

        means_no  = self.df.loc[self.df["DEFAULT_NUM"] == 0, bill_cols].mean()
        means_yes = self.df.loc[self.df["DEFAULT_NUM"] == 1, bill_cols].mean()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(month_lbls, means_no.values,  marker="o",
                color=self.COLORS["no_default"], linewidth=2, label="No Default")
        ax.plot(month_lbls, means_yes.values, marker="s",
                color=self.COLORS["default"],    linewidth=2, label="Default")

        ax.set_title("Average Monthly Bill Amount (Apr–Sep) by Default Status")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Bill Amount (NT$)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
        ax.legend()
        fig.tight_layout()
        return self._save_fig(fig, "fig6_avg_bill_over_time.png")

    # ------------------------------------------------------------------
    # Figure 7  –  Default rate vs number of consecutive delays
    # ------------------------------------------------------------------

    def fig_default_vs_delays(self, delay_df: pd.DataFrame):
        """
        Bar chart: default rate increases with number of months of delay.

        Parameters
        ----------
        delay_df : pd.DataFrame
            Output of DataAnalyzer.default_rate_by_delay_count().
        """
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(
            delay_df["CONSEC_DELAYS"],
            delay_df["default_rate_pct"],
            color=self.COLORS["highlight"],
            edgecolor="white",
        )

        for _, row in delay_df.iterrows():
            ax.text(
                row["CONSEC_DELAYS"],
                row["default_rate_pct"] + 1,
                f"{row['default_rate_pct']:.1f}%",
                ha="center", fontsize=9,
            )

        ax.set_title("Default Rate by Number of Months with Payment Delay ≥ 1")
        ax.set_xlabel("Number of Months with Delay")
        ax.set_ylabel("Default Rate (%)")
        ax.set_ylim(0, delay_df["default_rate_pct"].max() * 1.15)
        ax.set_xticks(delay_df["CONSEC_DELAYS"])
        fig.tight_layout()
        return self._save_fig(fig, "fig7_default_vs_delays.png")

    # ------------------------------------------------------------------
    # Figure 8  –  Payment ratio scatter
    # ------------------------------------------------------------------

    def fig_pay_ratio_scatter(self):
        """
        Scatter plot: utilisation rate vs payment ratio, coloured by default.
        Samples 3,000 points for legibility.
        """
        sample = self.df.sample(3000, random_state=42)
        no_def = sample[sample["DEFAULT_NUM"] == 0]
        def_   = sample[sample["DEFAULT_NUM"] == 1]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(no_def["UTIL_RATE"], no_def["PAY_RATIO"],
                   alpha=0.3, s=15, color=self.COLORS["no_default"], label="No Default")
        ax.scatter(def_["UTIL_RATE"],   def_["PAY_RATIO"],
                   alpha=0.3, s=15, color=self.COLORS["default"],    label="Default")

        ax.set_title("Credit Utilisation vs Payment Ratio (3,000 random clients)")
        ax.set_xlabel("Utilisation Rate (avg bill / credit limit)")
        ax.set_ylabel("Payment Ratio (avg payment / avg bill + 1)")
        ax.set_xlim(-0.1, 2.5)
        ax.set_ylim(-0.1, 3)
        ax.legend()
        fig.tight_layout()
        return self._save_fig(fig, "fig8_utilisation_vs_payment_ratio.png")

    # ------------------------------------------------------------------
    # Figure 9  –  Age distribution histogram
    # ------------------------------------------------------------------

    def fig_age_distribution(self):
        """Histogram of client ages with default overlay."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = range(int(self.df["AGE"].min()), int(self.df["AGE"].max()) + 2)

        ax.hist(self.df.loc[self.df["DEFAULT_NUM"] == 0, "AGE"],
                bins=bins, alpha=0.6, color=self.COLORS["no_default"],
                density=True, label="No Default")
        ax.hist(self.df.loc[self.df["DEFAULT_NUM"] == 1, "AGE"],
                bins=bins, alpha=0.6, color=self.COLORS["default"],
                density=True, label="Default")

        ax.set_title("Age Distribution by Default Status")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        return self._save_fig(fig, "fig9_age_distribution.png")

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _save_fig(self, fig, filename: str) -> str:
        """Save a figure and close it.  Returns the full path."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer] Saved → {path}")
        return path

    def generate_all_figures(self, analyzer) -> list:
        """
        Convenience method: generate and save all figures.

        Parameters
        ----------
        analyzer : DataAnalyzer
            Used to pull pre-computed statistics for certain plots.

        Returns
        -------
        list  Paths of saved figures.
        """
        corr   = analyzer.correlation_matrix()
        delays = analyzer.default_rate_by_delay_count()

        paths = [
            self.fig_default_distribution(),
            self.fig_credit_limit_histogram(),
            self.fig_default_by_education(),
            self.fig_default_by_age_group(),
            self.fig_correlation_heatmap(corr),
            self.fig_avg_bill_over_time(),
            self.fig_default_vs_delays(delays),
            self.fig_pay_ratio_scatter(),
            self.fig_age_distribution(),
        ]
        return [p for p in paths if p is not None]
