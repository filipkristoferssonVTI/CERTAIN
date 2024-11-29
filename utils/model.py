import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_model(model):
    pos = nx.circular_layout(model)
    nx.draw(model, pos=pos, with_labels=True)
    plt.draw()
    plt.show()


def cpd_to_df(cpd, col_node, row_node):
    return pd.DataFrame(cpd.get_values(), columns=cpd.state_names[col_node], index=cpd.state_names[row_node])
