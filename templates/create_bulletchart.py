import plotly.graph_objects as go

def create_bulletchart(data, layout_height=400, layout_margin={'t':0, 'b':0, 'l':0}):
    fig = go.Figure()
    
    for item in data:
        fig.add_trace(go.Indicator(
            mode = "number+gauge+delta",
            value = item.get('value', 0),
            delta = {'reference': item.get('delta_reference', 0)},
            domain = {'x': item.get('domain_x', [0.25, 1]), 'y': item.get('domain_y', [0, 1])},
            title = {'text': item.get('title', '')},
            gauge = {
                'shape': "bullet",
                'axis': {'range': [None, item.get('axis_range', 100)]},
                'threshold': {
                    'line': {'color': item.get('threshold_line_color', "black"), 'width': 2},
                    'thickness': 0.75,
                    'value': item.get('threshold_value', 0)},
                'steps': item.get('steps', []),
                'bar': {'color': item.get('bar_color', "black")}}))
    
    fig.update_layout(height=layout_height, margin=layout_margin)
    fig.show()

# Example Data
example_data = [
    {
        'value': 180,
        'delta_reference': 200,
        'domain_x': [0.25, 1],
        'domain_y': [0.08, 0.25],
        'title': "Revenue",
        'axis_range': 300,
        'threshold_value': 170,
        'steps': [{'range': [0, 150], 'color': "gray"}, {'range': [150, 250], 'color': "lightgray"}],
    },
    {
        'value': 35,
        'delta_reference': 50,
        'domain_x': [0.25, 1],
        'domain_y': [0.4, 0.6],
        'title': "Profit",
        'axis_range': 100,
        'threshold_value': 50,
        'steps': [{'range': [0, 25], 'color': "gray"}, {'range': [25, 75], 'color': "lightgray"}],
    },
    {
        'value': 220,
        'delta_reference': 210,
        'domain_x': [0.25, 1],
        'domain_y': [0.7, 0.9],
        'title': "Satisfaction",
        'axis_range': 300,
        'threshold_value': 210,
        'steps': [{'range': [0, 150], 'color': "gray"}, {'range': [150, 250], 'color': "lightgray"}],
    }
]

# Call the function with example data
create_bulletchart(example_data)
