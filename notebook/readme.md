# Notebooks

All exploratory and modelling notebooks live here, organised by **client type** and **modelling approach**.

## Structure

```
notebook/
├── new_clients/        # Work targeting new (first-time) clients
│   ├── freq_sev/       # Frequency–Severity approach
│   │   ├── dataset.ipynb
│   │   ├── explore.ipynb
│   │   ├── feature.ipynb
│   │   ├── frequency.ipynb
│   │   └── severity.ipynb
│   └── premium/        # Direct premium prediction approach
│       ├── datasets.ipynb
│       └── explore.ipynb
│
├── renewal_clients/    # Work targeting existing / renewal clients
│   └── feature.ipynb
│
└── readme.md
```

## Client Types

| Folder | Purpose |
|--------|---------|
| `new_clients/` | All exploration, feature engineering and modelling for **new clients** — customers we have no prior claims history for. |
| `renewal_clients/` | All exploration, feature engineering and modelling for **renewal clients** — existing customers where internal claims and payment history is available. |

## Modelling Approaches

The sub-folders within each client type represent a change in approach to arriving at the technical premium:

| Folder | Approach |
|--------|----------|
| `freq_sev/` | **Frequency–Severity** — models claims frequency and claims severity separately, then combines them to estimate the technical premium. |
| `premium/` | **Direct Premium** — predicts the technical premium directly in a single model. |

