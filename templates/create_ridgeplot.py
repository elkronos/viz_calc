import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_ridgeplot(df, x_column, row_column, title, palette="Set2", aspect=9, height=1.2, alpha=1, font_size=13, line_style='solid', line_color='black'):
    """
    This function creates a ridgeplot based on the inputs.
    
    # Existing parameter descriptions...
    
    font_size: int - The font size of the labels. Default is 13.
    line_style: str - The line style of the KDE plots. Default is 'solid'.
    line_color: str - The line color of the KDE plots. Default is 'black'.
    """
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
    palette = sns.color_palette(palette, df[row_column].nunique())
    
    g = sns.FacetGrid(df, row=row_column, hue=row_column, palette=palette, aspect=aspect, height=height)
    
    g.map_dataframe(sns.kdeplot, x=x_column, fill=True, alpha=alpha, lw=1.5, linestyle=line_style)
    g.map_dataframe(sns.kdeplot, x=x_column, color=line_color, lw=1)
    
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, color=color, fontsize=font_size, ha="left", va="center", transform=ax.transAxes)
    
    g.map(label, x_column)
    g.fig.subplots_adjust(hspace=-0.25)
    g.set_titles("")
    g.set(yticks=[])
    g.set_xlabels(x_column, fontsize=12)
    g.despine(left=True)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# Generate synthetic data with more variability
np.random.seed(0)
data = {
    'Category': [],
    'Value': []
}

for category in ['A', 'B', 'C', 'D', 'E']:
    values = np.random.normal(loc=np.random.randint(20, 100), scale=np.random.randint(5, 20), size=200)
    data['Category'].extend([category]*200)
    data['Value'].extend(values)

df = pd.DataFrame(data)

# Shuffle the data to make it more realistic
df = df.sample(frac=1).reset_index(drop=True)

# Use the create_ridgeplot function
create_ridgeplot(df, x_column='Value', row_column='Category', title='Enhanced Synthetic Ridgeplot Example')