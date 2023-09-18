import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_circularbar(data, id_col, group_col, value_col, individual_col, empty_bar=3, color_map="viridis", ylims=(-100, 120), alpha=0.5, value_lines=[20, 40, 60, 80], label_size=2.5, base_line_y=-5, group_label_y=-18):
    
    # Sorting and preparing data
    data = data.sort_values(by=[group_col, value_col])
    groups = data[group_col].unique()
    for group in groups:
        temp_df = pd.DataFrame({id_col: [None]*empty_bar, group_col: [group]*empty_bar, value_col: [None]*empty_bar, individual_col: [None]*empty_bar})
        data = pd.concat([data, temp_df], ignore_index=True)
    
    data = data.sort_values(by=[group_col, id_col])
    data['id'] = range(1, len(data)+1)

    # Preparing label data
    label_data = data.copy()
    number_of_bar = len(label_data)
    angle = 90 - 360 * (label_data['id']-0.5) / number_of_bar
    label_data['hjust'] = [1 if ang < -90 else 0 for ang in angle]
    label_data['angle'] = [ang+180 if ang < -90 else ang for ang in angle]
    
    # Preparing base data
    base_data = data.groupby(group_col)['id'].agg(['min', 'max'])
    base_data['max'] -= empty_bar
    base_data.reset_index(inplace=True)
    base_data['title'] = (base_data['min'] + base_data['max']) / 2

    # Plotting
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111, polar=True)
    ax.set_ylim(ylims)
    ax.set_yticklabels([])
    
    colors = plt.get_cmap(color_map, len(groups))
    
    for idx, group in enumerate(groups):
        group_data = data[data[group_col] == group]
        angles = (360 / number_of_bar) * (group_data['id'] - 0.5)
        values = group_data[value_col].fillna(0)
        width = 360 / number_of_bar - 0.2
        ax.bar(angles.apply(lambda x: x * (np.pi/180)), values, color=colors(idx), width=width*(np.pi/180), alpha=alpha)
        
        # Base line and group label
        start_angle, end_angle = (360 / number_of_bar) * (base_data.loc[idx, 'min'] - 0.5), (360 / number_of_bar) * (base_data.loc[idx, 'max'] - 0.5)
        theta = np.linspace(start_angle, end_angle, 100) * (np.pi/180)
        r = np.full((100,), base_line_y)
        ax.plot(theta, r, color='black', linewidth=1)
        ax.text(np.mean([start_angle, end_angle]) * (np.pi/180), group_label_y, group, ha='center', va='center', fontweight='bold')

    # Value lines and labels
    for val in value_lines:
        ax.plot([0, 360*(np.pi/180)], [val, val], color='grey', linewidth=0.3)
        ax.text(0, val, str(val), ha='right', va='center', color='grey', fontsize=8, fontweight='bold')
    
    # Individual labels
    for idx, row in label_data.iterrows():
        angle = (360 / number_of_bar) * (row['id'] - 0.5)
        value = row[value_col] if row[value_col] else 0
        hjust = row['hjust']
        ax.text(angle * (np.pi/180), value+10, row[individual_col], ha='left' if hjust == 0 else 'right', va='center', color='black', alpha=0.6, fontsize=label_size, fontweight='bold', rotation=row['angle'])
    
    plt.show()

# Usage example:
data = pd.DataFrame({
    'individual': ["Mister " + str(i+1) for i in range(60)],
    'group': ['A']*10 + ['B']*30 + ['C']*14 + ['D']*6,
    'value': list(np.random.choice(range(10, 100), 60, replace=True))
})

create_circularbar(data, 'index', 'group', 'value', 'individual')
