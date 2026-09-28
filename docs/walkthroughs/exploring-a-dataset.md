# Exploring a new dataset

**Goal:** get a first look at every column and every relationship, ranked
so the interesting ones come first, and hand the result to someone as a
slide deck.

```python
import viz_calc as vc
from viz_calc import datasets

df = datasets.measurements()   # 'group' and 'grade' are categorical; four numeric features
```

The three `profile_*` functions return pages of small multiples. `res.figure`
is a **list** of figures (12 panels per page by default), and `res.table`
summarizes every panel, so you can sort, filter or export it.

## 1. Value counts for every low-cardinality column

```python
res = vc.profile_bars(df, max_levels=10)
res.table.head()
```

A column is included when it has at most `max_levels` distinct values.
Missing values get their own "(missing)" bar rather than disappearing.

## 2. Every numeric column by every categorical column, strongest first

```python
res = vc.profile_boxes(df, max_levels=10)
res.table
```

![Profile boxes](../images/profile_boxes.png)

| categorical | numeric | n | eta_squared |
|---|---|---|---|
| group | depth | 150 | 0.94 |
| group | mass | 150 | 0.89 |
| group | length | 150 | 0.65 |
| group | width | 150 | 0.42 |
| grade | length | 150 | 0.03 |
| … | … | … | … |

Pairs are ranked by **η²** (eta-squared, column `eta_squared`), the share
of the numeric column's variance explained by the grouping. `group`
separates the specimens almost completely on depth; `grade` explains
practically nothing.

The original `all_boxes` script chose columns by whether their *maximum
value* was below 100 and paired numeric columns with each other. The
function now pairs categorical (few levels) with numeric (many levels)
columns, which is what a box plot compares.

## 3. Every numeric pair, strongest correlation first

```python
res = vc.profile_scatters(df, method="pearson", p_adjust="holm")
res.table[["x", "y", "r", "ci_low", "ci_high", "p_adjusted"]]
```

| x | y | r | ci_low | ci_high | p_adjusted |
|---|---|---|---|---|---|
| depth | mass | 0.96 | 0.94 | 0.97 | < .001 |
| length | mass | 0.90 | 0.87 | 0.93 | < .001 |
| length | depth | 0.87 | 0.82 | 0.90 | < .001 |
| width | depth | −0.44 | −0.56 | −0.30 | < .001 |
| width | mass | −0.41 | −0.53 | −0.27 | < .001 |
| length | width | −0.31 | −0.45 | −0.16 | < .001 |

Titles show *r* and the Holm-adjusted p-value. For large data the panels
draw a random sample of `max_points` rows, but the statistics use all rows.

## 4. Export

```python
res.save("scatter.png")   # one file per page: scatter_1.png, scatter_2.png, … (here all six pairs fit on one)
vc.to_pptx([vc.profile_bars(df), vc.profile_boxes(df), res],
           "first-look.pptx",
           titles=["Counts", "Numeric by category", "Numeric pairs"])
```

`to_pptx` (needs `pip install "viz_calc[pptx]"`) puts one figure per 16:9
slide, scaled to fit without distortion. The original scripts stretched each
image into a fixed 4.5 × 4.5 inch square and paused for keyboard input between
plots.
