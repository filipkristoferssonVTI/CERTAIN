from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    data_folder = Path('data/output/eda')
    df1 = pd.read_csv(data_folder / 'ärenden.csv')
    df2 = pd.read_csv(data_folder / 'grid.csv')

    merged_df = df1.merge(df2, on='rut_id')

    plt.figure(figsize=(19.2, 10.8))
    sns.boxplot(data=merged_df, x="handelse", y="POP")
    plt.xticks(rotation=90)
    plt.tight_layout()

    means = merged_df.drop(columns=['rut_id', 'POP']).groupby('handelse').mean()

    counts = merged_df['handelse'].value_counts()
    labels = [f"{h} (n={counts[h]})" for h in means.index]

    plt.figure(figsize=(19.2, 10.8))
    sns.heatmap(means.T, cmap="coolwarm")
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=90)
    plt.tight_layout()

    plt.show()

    # merged_df = pd.get_dummies(merged_df, columns=['handelse'], prefix='', prefix_sep='')
    # profile = ProfileReport(merged_df, title="EDA Report", explorative=True)
    # profile.to_file(data_folder / "eda_report.html")


if __name__ == '__main__':
    main()
