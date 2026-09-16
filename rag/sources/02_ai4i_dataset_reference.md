# AI4I 2020 dataset reference

## Provenance and limitation

The AI4I 2020 Predictive Maintenance Dataset is a synthetic dataset designed to
reflect predictive-maintenance data encountered in industry. It contains 10,000
observations and has no missing values according to the UCI repository. It is a
benchmark for this project, not a record of an identified physical machine or a
production maintenance system.

The dataset is released under the Creative Commons Attribution 4.0
International license. Cite the original source in project materials:

> AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine
> Learning Repository. https://doi.org/10.24432/C5HS5C

## Variables

The published data contains an identifier, product identifier, product quality
type, five operating measurements, an overall `Machine failure` target, and
five failure-mode labels.

| Variable | Meaning |
| --- | --- |
| `UDI` | Source row identifier; not a timestamp. |
| `Product ID` | Product-quality prefix (`L`, `M`, or `H`) and serial number. |
| `Type` | Low, medium, or high quality variant. |
| `Air temperature [K]` | Air temperature measurement. |
| `Process temperature [K]` | Process temperature measurement. |
| `Rotational speed [rpm]` | Rotational-speed measurement. |
| `Torque [Nm]` | Torque measurement. |
| `Tool wear [min]` | Tool-wear measurement. |
| `Machine failure` | Overall published failure label. |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Published failure-mode labels. |

## Modelling limitation

When predicting `Machine failure`, the five failure-mode labels are outcomes,
not sensor inputs. They must not be used as model features because that would
leak failure-related information. A prediction estimates the label learned from
this synthetic benchmark; it does not prove the cause of a failure.

## Source

- UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset:
  https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
- Retrieved on 2026-09-16.
