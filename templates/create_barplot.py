import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Optional

def to_title_case(string: str) -> str:
    """Convert a string to title case and replace underscores."""
    return string.replace('_', ' ').title()

def check_input(data: pd.DataFrame, x: str, y: Optional[str] = None, facet_col: Optional[str] = None) -> None:
    """Check if the given columns are present in the data frame."""
    if x not in data.columns or (y and y not in data.columns) or (facet_col and facet_col not in data.columns):
        raise ValueError(f"Column not found in data frame.")

def calculate_statistics(data: pd.DataFrame, x: str, y: Optional[str] = None, stat_type: Optional[str] = None, facet_col: Optional[str] = None) -> pd.DataFrame:
    """Calculate the required statistics for the bar plot."""
    data = data.copy()
    data[x] = data[x].astype(str).apply(to_title_case)
    
    if y:
        data[y].fillna(data[y].mean(), inplace=True)
        if facet_col:
            data_stat = data.groupby([x, facet_col], as_index=False).agg({y: stat_type or 'size'})
        else:
            data_stat = data.groupby(x, as_index=False).agg({y: stat_type or 'size'})
        data_stat[y] = data_stat[y].round(2)
    else:
        if facet_col:
            data_stat = data.groupby([x, facet_col]).size().reset_index(name='count')
        else:
            data_stat = data[x].value_counts().reset_index()
            data_stat.columns = [x, 'count']
        
    return data_stat

def create_barplot(data: pd.DataFrame, x: str, y: Optional[str] = None, stat_type: Optional[str] = None, 
                   show_labels: bool = True, x_labels: Optional[str] = None, y_labels: Optional[str] = None, 
                   show_title: bool = True, text_size: int = 12, bar_width: float = 0.8, 
                   color_palette: str = 'viridis', interactive: bool = False, 
                   plot_width: int = 10, plot_height: int = 6, facet_col: Optional[str] = None) -> None:
    """
    Create a bar plot using either seaborn or plotly, depending on the value of 'interactive'.
    ...
    """
    check_input(data, x, y, facet_col)
    data_stat = calculate_statistics(data, x, y, stat_type, facet_col)
    
    default_x_label = x_labels or x
    default_y_label = y_labels or (y or stat_type or 'count')
    default_title = f'{to_title_case(stat_type) if stat_type else "Frequency"} of {default_y_label} by {to_title_case(x)}'.strip()
    
    if interactive:
        fig = px.bar(data_stat, x=x, y=y if y else 'count', facet_col=facet_col, color=x, 
                     labels={x: default_x_label, y: default_y_label}, title=default_title if show_title else None, 
                     width=plot_width * 100, height=plot_height * 100, text=y if y and show_labels else None)
        fig.show()
    else:
        if facet_col:
            g = sns.FacetGrid(data_stat, col=facet_col, height=plot_height, aspect=plot_width/plot_height)
            g.map_dataframe(sns.barplot, x=x, y=y if y else 'count', palette=color_palette, width=bar_width)
            g.set_titles("{col_name}")
            g.set_axis_labels(default_x_label, default_y_label)
            
            if show_labels:
                for ax in g.axes.flat:
                    for patch in ax.patches:
                        ax.annotate(f"{patch.get_height():.2f}", 
                                    (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                                    ha='center', va='center', fontsize=text_size, color='black')
            plt.show()
        else:
            plt.figure(figsize=(plot_width, plot_height))
            sns.barplot(data=data_stat, x=x, y=y if y else 'count', palette=color_palette, width=bar_width)
            plt.xlabel(default_x_label)
            plt.ylabel(default_y_label)
            if show_title:
                plt.title(default_title)
            if show_labels:
                ax = plt.gca()
                for patch in ax.patches:
                    ax.annotate(f"{patch.get_height():.2f}", 
                                (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                                ha='center', va='center', fontsize=text_size, color='black')
            plt.show()


# Example data
data = pd.DataFrame({
    'category': ['a', 'b', 'a', 'c', 'b', 'c', 'a', 'b'],
    'value': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
    'facet_col': ['x', 'x', 'y', 'y', 'x', 'y', 'x', 'y']
})

# Example 1: Interactive plot without facet_col
create_barplot(data, 'category', 'value', 'mean', interactive=True)

# Example 2: Non-interactive plot without facet_col
create_barplot(data, 'category', 'value', 'mean', interactive=False)

# Example 3: Interactive plot with facet_col
create_barplot(data, 'category', 'value', 'mean', interactive=True, facet_col='facet_col')

# Example 4: Non-interactive plot with facet_col
create_barplot(data, 'category', 'value', 'mean', interactive=False, facet_col='facet_col')
