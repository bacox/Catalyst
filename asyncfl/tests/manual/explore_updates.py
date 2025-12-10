from pathlib import Path
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    file_loc = sys.argv[1]
    file_path = Path('.') / file_loc
    print(f'Loading data: {file_path}')
    with open(file_path, 'r') as infile:
        loaded_data = yaml.safe_load(infile)
        df = pd.DataFrame(loaded_data['updates'], columns=loaded_data['keys'])
        print(df)

        plt.figure()
        sns.scatterplot(data=df, x='server_age', y='global_conv', hue='byzantine')
        plt.show()
        plt.figure()
        sns.scatterplot(data=df, x='server_age', y='p_kt', hue='byzantine')
        plt.show()