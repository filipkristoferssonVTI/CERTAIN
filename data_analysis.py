from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


@dataclass
class Insats:
    insats: list[dict]

    @classmethod
    def from_group(cls, group):
        group = group[['Resurs, enhet',
                       'Resurs, tid larm',
                       'Resurs, tid kvittens',
                       'Resurs, tid klar',
                       'Resurs, tid framme',
                       'Resurs, tid avslut']]
        return cls(group.to_dict(orient='records'))


@dataclass
class Insatsbeslut:
    station: list[str]
    enhet: list[str]
    id: tuple[tuple[str, ...], tuple[str, ...]] = field(init=False)

    @classmethod
    def from_group(cls, group):
        return cls(group['Resurs, station'].tolist(), group['Resurs, enhet'].tolist())

    def __post_init__(self):
        self.station.sort()
        self.enhet.sort()
        assert len(self.station) == len(self.enhet)
        self.id = (tuple(sorted(self.station)), tuple(sorted(self.enhet)))

    def __hash__(self):
        # This ensures objects can be used in sets or as dictionary keys
        return hash(self.id)


@dataclass
class Arende:
    plats: str
    handelse: str
    arende: str
    insatsbeslut: Insatsbeslut
    insats: Insats


def arende_list_2_df(arende_list):
    return pd.DataFrame([{'plats': a.plats,
                          'handelse': a.handelse,
                          'arende': a.arende,
                          'insatsbeslut': a.insatsbeslut.id} for a in arende_list])


def check_column_uniformity(df, col):
    """Ignores nan."""
    non_nan_values = df[col].dropna()
    if non_nan_values.empty:
        return None
    unique_values = non_nan_values.unique()
    if len(unique_values) == 1:
        return unique_values[0]
    else:
        raise ValueError(f"The column '{col}' contains multiple unique non-NaN values: {unique_values}")


def no_duplicates_except_nan(group, col):
    if len(group) > 1:
        non_nan_values = group[col].dropna()
        return not non_nan_values.duplicated().any()
    return True


def group_data(df):
    groups = df.groupby('Ärende, årsnr')

    data = []

    for name, group in groups:
        plats = check_column_uniformity(group, 'Plats, miljö')
        handelse = check_column_uniformity(group, 'Händelse, typ')
        arende = check_column_uniformity(group, 'Ärende, förmodad händelse')

        # TODO: How do we want to handle dupl resurs-id in same group, what does this mean?
        insatsbeslut = Insatsbeslut.from_group(group)
        insats = Insats.from_group(group)

        data.append(Arende(plats=plats,
                           handelse=handelse,
                           arende=arende,
                           insatsbeslut=insatsbeslut,
                           insats=insats))

    return data


def clean_data(df):
    df['Resurs, station'] = df['Resurs, station'].astype(str)
    df['Resurs, enhet'] = df['Resurs, enhet'].astype(str)

    df['Resurs, tid larm'] = pd.to_datetime(df['Resurs, tid larm'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['Resurs, tid kvittens'] = pd.to_datetime(df['Resurs, tid kvittens'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['Resurs, tid klar'] = pd.to_datetime(df['Resurs, tid klar'], format='%Y-%m-%d %H:%M', errors='coerce')
    df['Resurs, tid framme'] = pd.to_datetime(df['Resurs, tid framme'], format='%Y-%m-%d %H:%M', errors='coerce')
    df['Resurs, tid avslut'] = pd.to_datetime(df['Resurs, tid avslut'], format='%Y-%m-%d %H:%M', errors='coerce')

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # strip leading and trailing whitespaces
    return df


def plot(data):
    kde = gaussian_kde(data)
    x_values = np.linspace(min(data), max(data), 1000)
    density = kde(x_values)

    plt.hist(data, bins='auto')
    plt.xlim(0, 60)
    plt.show()

    # KDE (Smooth Distribution)
    plt.plot(x_values, density)
    plt.show()


def main():
    data_folder = Path('data')
    df = pd.read_csv(data_folder / 'Kopia av Daedalos export - Insatta resurser 2201 2411.csv', sep=';')

    df = clean_data(df)
    data = group_data(df)
    data_df = arende_list_2_df(data)

    most_freq_insatsbeslut = data_df['insatsbeslut'].value_counts().idxmax()
    first_enhet_most_freq_insatsbeslut = most_freq_insatsbeslut[1][0]

    insatsbeslut_most_freq_first_enhet_tt = []
    for arende in data:
        if arende.insatsbeslut.id == most_freq_insatsbeslut:
            for item in arende.insats.insats:
                if item['Resurs, enhet'] == first_enhet_most_freq_insatsbeslut:
                    # TODO: Missing time is excluded, how do we handle these?
                    if not pd.isnull(item['Resurs, tid kvittens']) and not pd.isnull(item['Resurs, tid framme']):
                        travel_time = item['Resurs, tid framme'] - item['Resurs, tid kvittens']
                        travel_time = round(travel_time.seconds / 60)
                        insatsbeslut_most_freq_first_enhet_tt.append(travel_time)

    plot(insatsbeslut_most_freq_first_enhet_tt)

    # plats_handelse = pd.crosstab(data_df['plats'], data_df['handelse'])
    # arende_insatsbeslut = pd.crosstab(data_df['arende'], data_df['insatsbeslut'].astype(str))

    DEBUG = 1


if __name__ == '__main__':
    main()
