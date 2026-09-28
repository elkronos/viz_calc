"""Small, seeded synthetic datasets used by the documentation and tests.

They are generated, not downloaded, so the examples work offline and give
the same numbers every time. None of them describe real people or companies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["trial", "survey", "sales", "measurements", "projects", "network", "memberships"]


def trial(n_per_group: int = 40, seed: int = 1) -> pd.DataFrame:
    """A three-arm experiment: ``arm`` (control / low dose / high dose), ``site``, ``score``, ``baseline``.

    In the population the high dose raises the mean score by 6 points (about
    half a standard deviation); the low dose by 3 points with a larger
    spread, so the group variances differ and Welch's test matters.
    """
    rng = np.random.default_rng(seed)
    arms = {"control": (50, 10), "low dose": (53, 14), "high dose": (56, 10)}
    frames = []
    for arm, (mu, sd) in arms.items():
        baseline = rng.normal(50, 10, n_per_group)
        frames.append(pd.DataFrame({
            "arm": arm,
            "site": rng.choice(["North", "South"], n_per_group),
            "baseline": baseline.round(1),
            "score": (mu + 0.5 * (baseline - 50) + rng.normal(0, sd, n_per_group)).round(1),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = pd.Categorical(df["arm"], categories=list(arms), ordered=True)
    df["improved"] = (df["score"] > df["baseline"]).astype(int)
    return df


def survey(n: int = 300, seed: int = 2) -> pd.DataFrame:
    """Survey responses: five Likert items (1–5 labels), ``team``, ``tenure_years``, ``remote``."""
    rng = np.random.default_rng(seed)
    levels = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    leanings = {"Workload is manageable": -0.6, "My manager supports me": 0.8, "Tools are adequate": 0.1,
                "I see a future here": 0.4, "Meetings are useful": -1.0}
    df = pd.DataFrame({"team": rng.choice(["Data", "Design", "Engineering", "Sales"], n, p=[0.2, 0.15, 0.4, 0.25]),
                       "tenure_years": rng.gamma(2.0, 2.0, n).round(1),
                       "remote": rng.choice([True, False], n, p=[0.55, 0.45])})
    for item, lean in leanings.items():
        latent = lean + rng.normal(0, 1.1, n)
        df[item] = pd.cut(latent, [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf], labels=levels).astype(str)
    return df


def sales(seed: int = 3) -> pd.DataFrame:
    """Two years of daily sales for two channels, with weekly and yearly seasonality."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    t = np.arange(days.size)
    season = 1 + 0.25 * np.sin(2 * np.pi * (t - 80) / 365.25)
    weekly = np.where(days.weekday >= 5, 0.7, 1.05)
    online = 1000 * season * weekly * (1 + t / 1500) + rng.normal(0, 90, days.size)
    store = 1150 * season[::-1] * weekly + rng.normal(0, 90, days.size)
    return pd.DataFrame({"date": days, "online": online.round(0), "store": store.round(0),
                         "region": rng.choice(["East", "West"], days.size)})


def measurements(seed: int = 4) -> pd.DataFrame:
    """Numeric features for three species-like groups (for PCA, correlations, profiling)."""
    rng = np.random.default_rng(seed)
    specs = {"alpha": (5.0, 3.4, 1.5, 0.3), "beta": (5.9, 2.8, 4.3, 1.3), "gamma": (6.6, 3.0, 5.6, 2.0)}
    frames = []
    for g, (a, b, c, d) in specs.items():
        size = rng.normal(0, 1, 50)
        frames.append(pd.DataFrame({
            "group": g,
            "length": a + 0.4 * size + rng.normal(0, 0.25, 50),
            "width": b + 0.25 * size * (1 if g == "alpha" else -0.3) + rng.normal(0, 0.3, 50),
            "depth": c + 0.35 * size + rng.normal(0, 0.3, 50),
            "mass": d + 0.2 * size + rng.normal(0, 0.15, 50),
            "grade": rng.choice(["A", "B", "C"], 50),
        }))
    df = pd.concat(frames, ignore_index=True)
    num = ["length", "width", "depth", "mass"]
    df[num] = df[num].clip(lower=0.05).round(2)  # physical sizes cannot be negative
    return df


def projects() -> pd.DataFrame:
    """A small project plan: ``task``, ``team``, ``start``, ``end``, ``work_start``, ``work_end``."""
    return pd.DataFrame({
        "task": ["Discovery", "Design", "Build API", "Build UI", "Testing", "Launch"],
        "team": ["Research", "Design", "Engineering", "Engineering", "QA", "Ops"],
        "start": pd.to_datetime(["2024-01-08", "2024-01-29", "2024-02-19", "2024-03-04", "2024-04-15", "2024-05-13"]),
        "end": pd.to_datetime(["2024-02-02", "2024-03-01", "2024-04-19", "2024-04-26", "2024-05-17", "2024-05-24"]),
        "work_start": pd.to_datetime(["2024-01-10", "2024-02-05", "2024-02-26", "2024-03-11", "2024-04-22", "2024-05-15"]),
        "work_end": pd.to_datetime(["2024-01-31", "2024-02-23", "2024-04-12", "2024-04-19", "2024-05-10", "2024-05-22"]),
    })


def network(seed: int = 5) -> pd.DataFrame:
    """An edge list with two loosely linked friend groups: ``source``, ``target``, ``strength``."""
    rng = np.random.default_rng(seed)
    a = ["Ana", "Ben", "Cai", "Dee", "Eli", "Fay"]
    b = ["Gus", "Hal", "Ivy", "Jo", "Kai", "Lu"]
    edges = []
    for grp in (a, b):
        for i, u in enumerate(grp):
            for v in grp[i + 1:]:
                if rng.random() < 0.6:
                    edges.append((u, v, int(rng.integers(3, 10))))
    edges += [("Cai", "Hal", 2), ("Fay", "Gus", 1)]
    return pd.DataFrame(edges, columns=["source", "target", "strength"])


def memberships(seed: int = 6) -> dict[str, set[str]]:
    """Five overlapping sets of user IDs, e.g. users of five product features."""
    rng = np.random.default_rng(seed)
    users = [f"u{i:03d}" for i in range(400)]
    rates = {"Search": 0.6, "Export": 0.25, "Sharing": 0.35, "Alerts": 0.2, "API": 0.08}
    return {k: {u for u in users if rng.random() < p} for k, p in rates.items()}
