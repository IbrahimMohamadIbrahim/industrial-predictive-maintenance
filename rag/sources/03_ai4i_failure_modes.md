# Published AI4I failure-mode labels

The following definitions are specific to the synthetic AI4I 2020 benchmark.
They can explain the dataset labels, but they are not universal engineering
rules and must not be presented as a diagnosis of real equipment.

## TWF - tool-wear failure

In the data generator, a tool is selected for replacement or failure at a
randomly selected tool-wear time between 200 and 240 minutes. The published
dataset distinguishes replacement and failure events within that simulation.

## HDF - heat-dissipation failure

In the data generator, heat dissipation causes a process failure when the
process-to-air temperature difference is below 8.6 K and rotational speed is
below 1,380 rpm.

## PWF - power failure

In the data generator, process power is derived from torque and rotational
speed. A failure occurs when that power is below 3,500 W or above 9,000 W.

## OSF - overstrain failure

In the data generator, an overstrain failure occurs when the product of tool
wear and torque exceeds 11,000 minNm for type L, 12,000 minNm for type M, or
13,000 minNm for type H.

## RNF - random failure

The data generator includes a random failure mechanism independent of the
process parameters. It cannot be inferred deterministically from the listed
sensor values.

## Interpreting the labels

`Machine failure` is the overall published target. It is related to the
failure-mode labels in the source generator, but an ML prediction should not be
used to assert which mode caused a real-world failure. For model training,
these labels are excluded from the input features to avoid leakage.

## Source

- UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset,
  Additional Variable Information:
  https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
- Retrieved on 2026-09-16.
