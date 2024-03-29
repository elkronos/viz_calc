# viz_calc
 Visual calutators in python

## Analyses
Contains scripts designed to compute various analysis. These tend to be more specialized in terms of their design and purpose than templates.

- `all_bars` generates bar plots from columns in a pandas DataFrame and optionally saves them to a PowerPoint file. The function takes various parameters including the data to be plotted, maximum unique levels a column can have to be plotted, and customization options like bar color and label font size. Additionally, it defines a nested function set_plot_properties to set properties of the individual plots.

- `all_boxes` takes input data and several optional parameters to control the plot creation. The script includes input validation to ensure correct parameter types and ranges. The generated plots can either be displayed in groups or saved to a PowerPoint presentation, with relevant properties set through helper functions.

- `all_scatters` creates scatter plots for combinations of columns in a pandas DataFrame based on certain criteria. It includes functions to validate inputs, identify columns to plot based on unique values, create scatter plots (with options to include a line of best fit and a confidence interval), and set plot properties. The scatter plots can either be displayed or saved to a PowerPoint presentation.

- `animated_bubble` generates an interactive animated bubble plot from a pandas DataFrame, visualizing the evolution of data over time. It takes several parameters including the DataFrame and column names representing various plot elements like time, x-axis and y-axis values, bubble size and color, labels, as well as animation interval and bubble scaling factor. Utilizing matplotlib and FuncAnimation, the function iterates through unique time intervals, creating and updating a scatter plot with parameters specified by the data within each time slice, also showcasing a progress bar to indicate the loading status. 

- `bar_time` visualizes a time series dataset using a combination of bar and line plots. The function accepts a Pandas DataFrame data and parameters including date_col (the column containing date information), y_col (the column containing values to analyze), group_by (to group data by day, month, quarter, or year), window_size (to calculate moving statistics), and statistical measures bar_stat and line_stat to customize the visualization. It first validates the input parameters, then groups the data according to the specified frequency and calculates the specified statistics. The grouped data is then plotted with bars representing one statistical measure and a line plot showing the rolling average of another statistical measure.

- `bench_bar` takes a pandas DataFrame and several other parameters to generate a bar chart with error bars using Matplotlib. The bars can be colored based on a specified threshold, and error bars can represent either standard error or standard deviation. The function implements various error handling mechanisms to validate the input parameters before plotting the graph.

- `centered_barplot` takes a Pandas DataFrame data and visualizes the proportion of values in the column specified by y that are above and below a defined threshold, grouped by the categories specified in column x. If the threshold is not provided, it defaults to the mean of column y. The visualization is a centered bar plot, with bars above and below a central line representing the proportions above and below the threshold, respectively. The function allows customization such as adding labels on the bars and a title to the plot.

- `compare_correlations` offers a suite of functions to compare correlation matrices within groups in a dataset. Using either Pearson or Spearman correlation, the _compute_matrix function calculates the correlation matrix for a given subset of data. The _group_correlations function groups the data by a specified variable and computes their correlation matrices, subsequently determining the absolute differences between these matrices for each group combination. The _visualize_difference_matrices function visually represents these differences as heatmaps. The main function, compare_correlations, integrates these steps to compute and display correlations within groups. 

- `correlogram` generates a correlation heatmap from a pandas DataFrame using seaborn. The function allows the user to choose between Pearson's or Spearman's correlation methods, customize the color palette of the heatmap, and optionally annotate the cells with correlation values and/or significance indicators based on a specified p-value threshold. The script includes error handling to validate the input correlation method.

- `donut_charts` generates a series of donut charts from the columns of a given pandas DataFrame, optionally displaying the labels on the charts. It employs a dark-themed aesthetic, with a customizable color palette derived from the matplotlib viridis_r colormap, and dynamically arranges up to 8 charts in a 3x3 grid on a figure. Within the function, it iterates over the first 8 rows of the DataFrame, creating pie charts with a donut aesthetic for each row, and integrates the option to either show or hide data labels. The function also features a legend detailing the categories present in the DataFrame and adjusts the spacing between charts for better visualization.

- `divergent_bar` visualizes the comparison between two different sets of data (specified by col_left and col_right) across various categories (specified by col_cat) using a horizontal divergent bar plot. It receives a Pandas DataFrame data and various other parameters to customize the plot, including the colors of the bars, labels for the left and right sides, and an optional title. The data from the specified columns are extracted, combined, and sorted based on the category column to create a structured dataset for plotting. The function then constructs a divergent bar plot where one set of data is represented as negative values on the left side and the other set as positive values on the right side. The user can optionally save the plot to a file by providing a filename to the save_to_file parameter.

- `duration_plot` uses pandas and Matplotlib to create horizontal bar charts visualizing time durations for different groups. It takes parameters for start and end dates of an overall period and start and end dates for a specific duration within that period, along with a grouping variable. The function sorts the data by start date (earliest to latest), converts date columns to datetime objects, and dynamically adjusts the x-axis format based on the range of dates. It plots two sets of bars for each group: one representing the overall period and another indicating a specific duration within that period. The function also includes gridlines, custom date formatting for the x-axis depending on the date range, and a legend, with automatic adjustment for figure size based on the number of groups.

- `grid_pecent` creates a visualization of percentages in a grid format, particularly useful for displaying binary data (like 0 and 1). This function takes in a Pandas DataFrame data and a specified column_name which should contain binary data to calculate percentages. You can specify various options, including whether to display labels (label), the colors to use for the grid (colors), and an optional facet variable (facet_var) to create separate grids for different subgroups within the data, with a potential custom order (facet_order).

- `heatmap_calendar` plots a series of calendar heatmaps for each year within a specified date range, visualizing the data from the date_values parameter. It accepts various parameters to customize the heatmap's appearance, such as colormap (cmap), fill color (fillcolor), line width (linewidth), and line color (linecolor). The function uses the pandas library to organize the data into a time series and the calmap library to plot individual yearly calendar heatmaps within a loop that iterates from the start to the end year derived from the start_date parameter. It also incorporates error handling to manage any exceptions that occur during execution.

- `histo_group` generates histograms or density plots from a pandas DataFrame to analyze the distribution of a numerical variable, potentially grouped and faceted by other categorical variables. The function allows customization of the visualization with various parameters, including bin width, overlaying density plots, adding statistical lines (like mean or median), and modifying aesthetic elements such as title and labels. The example at the end demonstrates the use of the function with a synthesized dataset, illustrating the distribution of 'mpg' values grouped by 'type' and faceted by 'gear'.

- `lollipop_plot` visualizes data from a pandas DataFrame as a lollipop chart, where it groups the data by the x_var column and calculates the specified statistic ('mean', 'median', or 'sum') for the y_var column. The resulting aggregated values are plotted as lollipops, with optional customization such as changing color, adjusting the size of the dots, and adding labels with abbreviated numbers. A helper function, abbreviate_number, is used to format the labels with appropriate suffixes ('K' for thousands, 'M' for millions, etc.). A threshold line can also be added to the plot to highlight values above or below a particular level.

- `nested_pie` provides a pie chart with sub-categories nested within the primary category from a pandas DataFrame. This function is for visualizing data with both inner and outer categories represented in concentric rings. It takes several parameters, including the DataFrame to be plotted, chart title, colormap name, figure size, edge color, line width, and optional labels for both inner and outer categories. Additionally, it provides an option to save the generated chart as a file with the specified path in the 'save_as' parameter. The function includes error handling to ensure that the input data is a pandas DataFrame and utilizes the matplotlib library to construct and display the chart.

- `network_map` function generates an interactive network graph from input data, typically in the form of a Pandas DataFrame. Users can specify source and target node columns, edge weight, node size, and color columns for customization. Additional options include choosing a colorscale, enabling community detection, and interactivity. The function uses NetworkX to construct the graph, applies Louvain community detection if requested, and utilizes a spring layout algorithm for node positioning. It visualizes edges and nodes using Plotly, allowing customization of color, size, and hover information. The resulting graph is displayed with options for layout, legend, and hover settings. Users can access the Python code via a Plotly link in the graph.

- `pairs_plots` generates a pair plot of specified variables from a dataset, which can be loaded using a URL or filepath. It offers various customization options including plot kind (plot_kind), color palette (palette), and markers style (markers). Other parameters allow control over the size (size) and aspect ratio (aspect) of the plot facets, the kind of plot on the diagonal (diag_kind), and whether to exclude the diagonal plots (corner). The function also has options to apply a log-scale transformation to the variables (log_scale) and to save the plot to a file (save_as). If return_grid is set to True, the function returns a PairGrid instance for further customization. It contains error handling to catch and report any exceptions that occur during its execution.

- `plot_pca` conducts PCA on input data and plots the results as a scatterplot, using seaborn to differentiate groups with color. It includes circles around each group, with size and orientation determined by PCA. The function also adds group labels to the circles, and the plot has customizable aesthetics, axis labels, a title, and a legend.

- `quadrant_norm` creates a scatter plot with options to display Pearson's correlation coefficient, annotate quadrants with percentages, and save the plot. It accepts data from a pandas DataFrame and rescales the specified x and y columns for standardized plotting. The function allows for customization of labels, colors, and font styles. It can be used interactively or to save the plot to a file. The example at the end demonstrates its usage with synthetic data, showcasing scatter plots with various customizations.

- `significant_means` plots the mean values of two groups with their standard error of the mean (SEM) using the matplotlib library. It also performs a t-test to ascertain if the difference between the means of the two groups is significant using the scipy stats module. The function visualizes the data with bar plots indicating the mean values and the errors and annotates the plot with the p-value from the t-test, highlighting if there's a significant difference between the groups based on a specified alpha level.

- `stacked_percentages` plots a stacked bar chart showing percentages of one categorical variable within another. It computes percentages based on counts and provides customization options for color, labels, and plot appearance. The example at the end demonstrates its usage with a sample dataset, visualizing the percentages of 'cut' categories within 'color' categories.

- `timeseries_fill` creates a time series plot from a pandas DataFrame. It groups the data by the time_col column and plots the specified series columns (series_cols). It can optionally fill the area between two series if fill_between is set to True. You can customize the colors, title, x-axis label, y-axis label, alpha value, and line width. The function is demonstrated using synthetic data, visualizing crossovers between two series.

## Templates
Contains scripts designed to make general plots. Offers options to edit aesthetics, wrap in plotly, as well as facet or group by variables where applicable.

- `create_barplot`
- `create_boxjitter`
- `create_boxplot`
- `create_bubbleplot`
- `create_choropleth`
- `create_circularbar`
- `create_densityplot`
- `create_dumbellplot`
- `create_funnelplot`
- `create_gantt`
- `create_heatmap`
- `create_histogram`
- `create_markermap`
- `create_radarplot`
- `create_ridgeplot`
- `create_sankeydiagram`
- `create_scatterplot`
- `create_stockplot`
- `create_venn`
- `create_violinplot`
- `create_waffle`
- `create_waterfallchart`
- `create_wordcloud`
