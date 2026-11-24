# Gallery

Every chart below is produced by `examples/gallery.py`; this page is generated from it.
All examples use this setup:

```python
import viz_calc as vc
from viz_calc import datasets
import pandas as pd

trial = datasets.trial()
survey = datasets.survey()
sales = datasets.sales()
meas = datasets.measurements()
proj = datasets.projects()
edges = datasets.network()
survey["work_mode"] = survey["remote"].map({True: "Remote", False: "On-site"})
likert_items = ["Workload is manageable", "My manager supports me", "Tools are adequate",
                "I see a future here", "Meetings are useful"]
levels = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
monthly = (sales.assign(month=sales["date"].dt.to_period("M").astype(str))
           .groupby("month", as_index=False)[["online", "store"]].sum())
# made-up Gapminder-style data: six countries every five years (GDP per person in $1,000s, population in millions)
start = pd.DataFrame({"country": ["A", "B", "C", "D", "E", "F"],
                      "continent": ["Africa", "Africa", "Asia", "Asia", "Europe", "Europe"],
                      "gdp": [1.5, 3.0, 4.0, 9.0, 25.0, 35.0], "life_exp": [55.0, 60.0, 66.0, 71.0, 77.0, 80.0],
                      "population": [30.0, 60.0, 90.0, 140.0, 10.0, 70.0]})
countries = pd.concat([start.assign(year=year, gdp=start["gdp"] * 1.04 ** (year - 2000),
                                    life_exp=start["life_exp"] + 0.25 * (year - 2000),
                                    population=start["population"] * 1.01 ** (year - 2000))
                       for year in range(2000, 2021, 5)], ignore_index=True)
```

## Comparing groups

### `estimation_plot`

```python
res = vc.estimation_plot(trial, x="arm", y="score")
```

![estimation_plot](images/estimation_plot.png)

### `benchmark_bar`

```python
res = vc.benchmark_bar(trial, x="arm", y="score", threshold=50)
```

![benchmark_bar](images/benchmark_bar.png)

### `lollipop`

```python
res = vc.lollipop(survey, x="team", y="tenure_years", stat="median", threshold=3)
```

![lollipop](images/lollipop.png)

### `dumbbell`

```python
res = vc.dumbbell(monthly.tail(8), label="month", start="store", end="online",
    sort=False, show_values=True)
```

![dumbbell](images/dumbbell.png)

### `divergent_bar`

```python
res = vc.divergent_bar(monthly.tail(8), category="month", left="store", right="online")
```

![divergent_bar](images/divergent_bar.png)

### `centered_bar`

```python
res = vc.centered_bar(trial, x="arm", y="score", threshold=55)
```

![centered_bar](images/centered_bar.png)

### `likert`

```python
res = vc.likert(survey, items=likert_items, levels=levels)
```

![likert](images/likert.png)

## Correlation

### `correlogram`

```python
res = vc.correlogram(meas, method="spearman")
```

![correlogram](images/correlogram.png)

### `compare_correlations`

```python
res = vc.compare_correlations(meas, group="group", columns=["length", "width", "depth", "mass"])
```

![compare_correlations](images/compare_correlations.png)

### `quadrant_plot`

```python
res = vc.quadrant_plot(trial, x="baseline", y="score")
```

![quadrant_plot](images/quadrant_plot.png)

## Distributions

### `histogram`

```python
res = vc.histogram(trial, x="score", hue="arm", facet="site", stat="density", ref_line="mean")
```

![histogram](images/histogram.png)

### `ridgeplot`

```python
res = vc.ridgeplot(meas, x="depth", group="group")
```

![ridgeplot](images/ridgeplot.png)

### `raincloud`

```python
res = vc.raincloud(trial, x="arm", y="score")
```

![raincloud](images/raincloud.png)

## Part-to-whole, flows and sets

### `waffle`

```python
res = vc.waffle(survey, category="team")
```

![waffle](images/waffle.png)

### `percent_grid`

```python
res = vc.percent_grid(trial, column="improved", facet="arm")
```

![percent_grid](images/percent_grid.png)

### `stacked_percentages`

```python
res = vc.stacked_percentages(survey, group="team", category="Meetings are useful",
    category_order=levels, colors="RdBu", horizontal=True)
```

![stacked_percentages](images/stacked_percentages.png)

### `donut_grid`

```python
res = vc.donut_grid(pd.crosstab(survey["team"], survey["work_mode"]))
```

![donut_grid](images/donut_grid.png)

### `nested_pie`

```python
res = vc.nested_pie(survey, outer="team", inner="work_mode")
```

![nested_pie](images/nested_pie.png)

### `circular_bar`

```python
res = vc.circular_bar(monthly.assign(year=monthly["month"].str[:4]), label="month",
    value="online", group="year", sort=False)
```

![circular_bar](images/circular_bar.png)

### `waterfall`

```python
res = vc.waterfall(pd.DataFrame({"item": ["Opening", "Sales", "Refunds", "Costs", "Tax"],
                  "amount": [1000, 650, -120, -380, -45]}),
    label="item", value="amount", start_label="Opening", total_label="Closing")
```

![waterfall](images/waterfall.png)

### `funnel`

```python
res = vc.funnel(pd.DataFrame({"stage": ["Visited", "Signed up", "Activated", "Paid"],
    "users": [10000, 3200, 1900, 410]}), stage="stage", value="users")
```

![funnel](images/funnel.png)

### `bullet`

```python
res = vc.bullet(pd.DataFrame({"kpi": ["Revenue", "Profit", "Satisfaction"],
                  "value": [270, 22, 4.1], "target": [250, 26, 4.5],
                  "poor": [150, 10, 2.5], "ok": [225, 20, 3.5], "good": [300, 30, 5]}),
    label="kpi", value="value", target="target", bands=["poor", "ok", "good"])
```

![bullet](images/bullet.png)

### `upset`

```python
res = vc.upset(datasets.memberships(), max_intersections=15)
```

![upset](images/upset.png)

## Time and projects

### `period_bars`

```python
res = vc.period_bars(sales, date="date", value="online", freq="month", stat="sum")
```

![period_bars](images/period_bars.png)

### `timeseries_fill`

```python
res = vc.timeseries_fill(sales.set_index("date")[["online", "store"]].resample("W").mean()
    .reset_index(), time="date", series=["online", "store"])
```

![timeseries_fill](images/timeseries_fill.png)

### `calendar_heatmap`

```python
res = vc.calendar_heatmap(sales, date="date", value="online")
```

![calendar_heatmap](images/calendar_heatmap.png)

### `gantt`

```python
res = vc.gantt(proj, task="task", start="start", end="end", group="team")
```

![gantt](images/gantt.png)

### `duration_plot`

```python
res = vc.duration_plot(proj, label="task", start="start", end="end",
    inner_start="work_start", inner_end="work_end")
```

![duration_plot](images/duration_plot.png)

### `animated_bubble`

```python
res = vc.animated_bubble(countries, time="year", x="gdp", y="life_exp", size="population",
    color="continent", label="country")
```

![animated_bubble](images/animated_bubble.png)

The image is the first of five frames; `res.info["animation"]` plays them all.

## Multivariate and networks

### `pca_plot`

```python
res = vc.pca_plot(meas, features=["length", "width", "depth", "mass"], group="group", loadings=True)
```

![pca_plot](images/pca_plot.png)

### `radar`

```python
res = vc.radar(meas, metrics=["length", "width", "depth", "mass"], group="group")
```

![radar](images/radar.png)

### `network_map`

```python
res = vc.network_map(edges, source="source", target="target", weight="strength", size_by="betweenness")
```

![network_map](images/network_map.png)

## Exploring a dataset

### `profile_bars`

```python
res = vc.profile_bars(meas, max_levels=10)
```

![profile_bars](images/profile_bars.png)

### `profile_boxes`

```python
res = vc.profile_boxes(meas, max_levels=10)
```

![profile_boxes](images/profile_boxes.png)

### `profile_scatters`

```python
res = vc.profile_scatters(meas, method="pearson", p_adjust="holm")
```

![profile_scatters](images/profile_scatters.png)
