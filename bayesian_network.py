from pathlib import Path

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


def filter_data(df):
    # TODO: Remove this.
    df = df.loc[df['Händelse, uppdrag'] == 'Brand']

    df = df[['Ärende, årsnr', 'Plats, miljö', 'Ärende, förmodad händelse', 'Händelse, typ']]

    print(len(df))
    df = df.dropna(subset=['Ärende, förmodad händelse', 'Händelse, typ'])
    print(len(df))

    # TODO: Each row is a vehicle? If not - use 'Resurs, enhet' to determine nr of vehicles.
    vehicle_count = df.groupby('Ärende, årsnr').size().reset_index(name='vehicles')
    representative_rows = df.groupby('Ärende, årsnr').first().reset_index()
    df = pd.merge(representative_rows, vehicle_count, on='Ärende, årsnr', how='left')

    df['Ärende, årsnr'] = df['Ärende, årsnr'].astype(str)
    df['Plats, miljö'] = df['Plats, miljö'].astype(str).astype('category')
    df['Ärende, förmodad händelse'] = df['Ärende, förmodad händelse'].astype(str).astype('category')
    df['Händelse, typ'] = df['Händelse, typ'].astype(str).astype('category')
    df['vehicles'] = df['vehicles'].astype(int)

    df['vehicle_bins'] = pd.cut(df['vehicles'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, max(df['vehicles'])])

    df = df.drop(columns=['Ärende, årsnr', 'Plats, miljö', 'vehicles'])
    return df


def main():
    data_folder = Path('data')
    df = pd.read_csv(data_folder / 'Kopia av Daedalos export - Insatta resurser 2201 2411.csv', sep=';')

    df = filter_data(df)

    # TODO: Are these reasonable dependencies?
    model = BayesianNetwork([('Händelse, typ', 'Ärende, förmodad händelse'),
                             ('Ärende, förmodad händelse', 'vehicle_bins')])

    model.fit(df, estimator=MaximumLikelihoodEstimator)

    inference = VariableElimination(model)  # noqa
    evidence_1 = {'Händelse, typ': 'Matlagning utan risk för skada'}
    evidence_2 = {'Händelse, typ': 'Brand eller brandtillbud i fordon eller fartyg utomhus'}

    prob_1 = inference.query(variables=['Ärende, förmodad händelse', 'vehicle_bins'], evidence=evidence_1)
    prob_2 = inference.query(variables=['Ärende, förmodad händelse', 'vehicle_bins'], evidence=evidence_2)
    prob_3 = inference.query(variables=['vehicle_bins'], evidence=evidence_1)
    prob_4 = inference.query(variables=['vehicle_bins'], evidence=evidence_2)

    print(prob_1)
    print('\n--------\n')
    print(prob_2)
    print('\n--------\n')
    print(prob_3)
    print('\n--------\n')
    print(prob_4)

    # plot_model(model)

    # cpd = model.get_cpds('vehicle_bins')
    # cpd = cpd_to_df(cpd, 'Ärende, förmodad händelse', 'vehicle_bins')
    # cpd.to_csv(data_folder / 'output' / 'cpd_test.csv')


if __name__ == '__main__':
    main()
