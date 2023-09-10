import plotly.graph_objects as go

def create_funnelplot(data, interactive=True):
    """
    Creates a funnel plot using Plotly, with an option to make it interactive.

    Parameters:
    data (dict): A dictionary containing the stage names as keys and the number of items at that stage as values.
    interactive (bool): Whether to make the plot interactive or not. Defaults to True.

    Returns:
    plotly.graph_objects.Figure: A plotly figure object which can be displayed or saved as an HTML file.
    """
    
    # Extract the stage labels and values
    stage_labels = list(data.keys())
    stage_values = list(data.values())

    # Create the funnel plot
    fig = go.Figure()

    fig.add_trace(go.Funnel(
        y=stage_labels,
        x=stage_values,
        textinfo="value+percent initial",
    ))

    # Disable interactivity if the interactive parameter is set to False
    if not interactive:
        fig.update_traces(hoverinfo='none', hovertemplate=None)
    
    return fig

# Example usage
data_example = {"Stage 1": 1000, "Stage 2": 500, "Stage 3": 200, "Stage 4": 100}

# To create and display an interactive funnel plot, the user only needs to call the function with the data dictionary
create_funnelplot(data_example).show()

# To create and display a non-interactive funnel plot, the user can specify the interactive parameter as False
create_funnelplot(data_example, interactive=False).show()
