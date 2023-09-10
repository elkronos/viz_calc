import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Optional

def to_title_case(string: str) -> str:
    return string.replace('_', ' ').title()

def check_input(data: pd.DataFrame, x: str, y: Optional[str]) -> None:
    if x not in data.columns or (y and y not in data.columns):
        raise ValueError(f"Column '{x if x not in data.columns else y}' not found in data frame.")

def create_barplot(data: pd.DataFrame, x: str, y: Optional[str] = None, stat_type: Optional[str] = None, 
             show_labels: bool = True, x_labels: Optional[str] = None, y_labels: Optional[str] = None, 
             show_title: bool = True, text_size: int = 12, bar_width: float = 0.8, 
             color_palette: str = 'viridis', interactive: bool = False) -> None:
    """
    Creates a bar plot with various customization options.
    
    Parameters:
    data (pd.DataFrame): Data frame containing the data.
    x (str): Name of the categorical variable.
    y (str, optional): Name of the continuous variable. Defaults to None.
    stat_type (str, optional): Summary statistic type ("sum", "mean", "median", "min", "max"). Defaults to None.
    show_labels (bool, optional): Whether to show labels on the bars. Defaults to True.
    x_labels (str, optional): Label for the x-axis. Defaults to None.
    y_labels (str, optional): Label for the y-axis. Defaults to None.
    show_title (bool, optional): Whether to show title. Defaults to True.
    text_size (int, optional): Text size for the labels and title. Defaults to 12.
    bar_width (float, optional): Width of the bars. Defaults to 0.8.
    color_palette (str, optional): Seaborn color palette to use. Defaults to 'viridis'.
    interactive (bool, optional): Whether to use interactive plotly graphs. Defaults to False.
    
    Returns:
    None
    """
    
    stat_functions = ["sum", "mean", "median", "min", "max"]
    
    check_input(data, x, y)

    if stat_type and stat_type not in stat_functions:
        raise ValueError(f"Invalid stat_type '{stat_type}'. Choose one of the following: {', '.join(stat_functions)}.")

    data = data.copy()
    data[x] = data[x].astype(str).apply(to_title_case)
    
    if y:
        data[y].fillna(data[y].mean(), inplace=True)
        if stat_type:
            data_stat = data.groupby(x, as_index=False).agg({y: stat_type})
        else:
            data_stat = data.groupby(x, as_index=False).agg({y: 'size'})
        data_stat[y] = data_stat[y].round(2)
    else:
        data_stat = data[x].value_counts().reset_index()
        data_stat.columns = [x, 'count']
    
    if interactive:
        color_discrete_sequence = sns.color_palette(color_palette, len(data_stat)).as_hex()
        fig = px.bar(data_stat, x=x, y=y if y else 'count', text=y if y else 'count', 
                     labels={x: x_labels if x_labels else to_title_case(x), 
                             y if y else 'count': y_labels if y_labels else (to_title_case(stat_type) if stat_type else 'Count')},
                     title=f'{to_title_case(stat_type) if stat_type else "Count"} of {to_title_case(y) if y else ""} by {to_title_case(x)}'.strip() if show_title else None,
                     color_discrete_sequence=color_discrete_sequence)
        fig.show()
    else:
        plt.figure()
        plot = sns.barplot(x=x, y=y if y else 'count', data=data_stat, errcolor='none', palette=color_palette, width=bar_width)
        
        if show_labels:
            for index, row in data_stat.iterrows():
                plot.text(row.name, row[y] if y else row['count'], round(row[y], 2) if y else round(row['count'], 2), color='black', ha="center", fontsize=text_size-2)
        
        if show_title:
            title = f'{to_title_case(stat_type) if stat_type else "Count"} of {to_title_case(y) if y else ""} by {to_title_case(x)}'.strip()
            plt.title(title, fontsize=text_size)
        
        plt.xlabel(x_labels if x_labels else to_title_case(x), fontsize=text_size)
        plt.ylabel(y_labels if y_labels else (to_title_case(stat_type) if stat_type else 'Count'), fontsize=text_size)
        plt.xticks(rotation=30, ha='right', fontsize=text_size-1)
        plt.tight_layout()
        plt.show()

# Example usage
data = pd.DataFrame({
    'category': ['a', 'b', 'a', 'c', 'b', 'c', 'a', 'b'],
    'value': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
})

create_barplot(data, 'category', 'value', 'mean', interactive=True)
