try:
    import matplotlib
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
from src.data import load_data 


def visualize_datasets():
    datasets = {
        x: os.path.join('data', x)
        for x in os.listdir('data') if x.endswith('.csv')
    }

    plottable = {}
    for name, path in datasets.items():
        x, y, attrs = load_data(path)
        if len(attrs) == 2:
            plottable[name] = (x, y)

    fig, axes = plt.subplots(nrows=1, ncols=len(plottable),
                             figsize=(4 * len(plottable), 4))
    for idx, name in enumerate(sorted(plottable.keys())):
        x, y = plottable[name]
        axes[idx].scatter(x[:, 0], x[:, 1], c=y)
        axes[idx].set_title(name.replace(".csv", ""))

    plt.tight_layout()
    plt.show()
    plt.close('all')


if __name__ == "__main__":
    visualize_datasets()
