import pandas as pd
import matplotlib.pyplot as plt

def divergent_bar(data, col_left, col_right, col_cat, title=None, 
                  left_color="#0072B2", right_color="#F0E442", 
                  left_label="Left", right_label="Right",
                  save_to_file=None):
    # Subset data into left and right columns
    data_left = data[col_left]
    data_right = data[col_right]
    data_cat = data[col_cat]
    
    # Combine the data into a single data frame
    combined_data = pd.DataFrame({
        "Category": list(data_cat) * 2,
        "Type": [left_label] * len(data) + [right_label] * len(data),
        "Value": list(-data_left) + list(data_right)  # Negate the values on the left side
    })
    
    # Sort the data based on the categorical variable
    combined_data = combined_data.sort_values(by='Category', ascending=False)
    
    # Create a divergent bar plot using matplotlib
    plt.figure(figsize=(10,6))
    for category in combined_data['Category'].unique():
        subset = combined_data[combined_data['Category'] == category]
        plt.barh(category, subset[subset['Type'] == left_label]['Value'], color=left_color)
        plt.barh(category, subset[subset['Type'] == right_label]['Value'], color=right_color)
    
    plt.xlabel('Value')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x')
    plt.legend([left_label, right_label], loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)
    
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches='tight')
    else:
        plt.show()

# Synthetic data creation and function testing
data = pd.DataFrame({
    'customers': list('ABCDEFGHIJ'),
    'market_campaign_a': [100, 120, 140, 160, 180, 150, 130, 110, 170, 190],
    'market_campaign_b': [50, 40, 30, 70, 60, 55, 65, 75, 85, 45]
})

# Use the `divergent_bar` function with the synthetic data
divergent_bar(data, 'market_campaign_b', 'market_campaign_a', 'customers', 
              title="Comparison of Campaigns A and B", 
              left_color="#F39C12", right_color="#2E9BDA",
              left_label="Method A", right_label="Method B")
