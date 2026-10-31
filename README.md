# Saâd Qacif

**AI engineer working across mathematical modelling, statistical inference, and sequential decision-making.**

I build research-driven AI systems for decisions under uncertainty. My interests span graph reinforcement learning, stochastic modelling, scientific machine learning, and quantitative research. I was ranked first in my MITx Statistics and Data Science MicroMasters cohort and in AI Engineering at ENSIAS.

## Featured research · Graph RL for soccer-player tracking

**[Law-of-Large-Numbers-Based Graph Deep RL](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking)** studies how to recover missing soccer players from noisy frame-by-frame detections without improving a tracking metric by simply deleting uncertain detections.

The project explores a set-to-set approach: input-conditioned slot attention proposes player positions and activity decisions; policy-gradient experiments use set-level recall and localization feedback. A law-of-large-numbers argument motivates learning from varied sequences to reduce sequence-specific bias. The repository also contains matching utilities, evaluation scripts, tests, and dated experiment notes.

The application is soccer tracking, not financial markets. Its relevance to quantitative research lies in the underlying mathematics: stochastic state estimation, set-valued prediction, assignment, and sequential optimization under noisy observations.

**Inspect the work:** [training implementation](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/blob/main/scripts/train_graph_rl_body_completion.py) · [matching code](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/matching) · [tests](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/tests) · [experiment notes](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/sessions)

The public repository is a code-only research extract; licensed media, source data, generated outputs, and trained weights are not redistributed. Cross-sequence transfer remains a research question rather than a result established by this public extract.

## Featured research · Decision-aware financial information transport

**[fininfo](https://github.com/SaadQacif/dynamax/tree/Saad-The-Financial-Markets-Alchemist/fininfo)**, built as an extension of [Dynamax](https://github.com/probml/dynamax), implements the draft *Decision-Aware Financial Information Transport in State-Space Models*. It asks what a piece of news or data is worth to a financial decision. Instead of scoring a packet by similarity, it models the packet as a statistical experiment with a likelihood. It then carries the resulting posterior update through state-space dynamics into payoff revisions, decision values with explicit units, and attributions that account for dependence.

The JAX kernels cover:

- exact Gaussian transport, with information and canonical-mode bounds;
- the value of information under constrained quadratic trading, and an exact CARA-Gaussian cash fee;
- residualization of correlated packets;
- Shapley packet attribution, with a counterexample showing the objective is not submodular;
- integrated-gradient attribution over transition edges;
- discounted cash-flow projection;
- source-reliability mixtures with a general null density.

Each kernel is tested against an independent route: joint-Gaussian Schur complements, SciPy densities, high-order quadrature or Monte Carlo simulation. Integration tests run against real `LinearGaussianSSM.filter` outputs.

In the synthetic misspecification studies, the correctly specified models give 95% intervals that cover 94.97% (correlated packets) and 95.13% (unreliable source). Ignoring the packet correlation drops coverage to 91.15%, and always trusting an unreliable source drops it to 72.54%. These results check the method against its own assumptions; they are not evidence of market predictability.

**Inspect the work:** [method and theorems](https://github.com/SaadQacif/dynamax/blob/Saad-The-Financial-Markets-Alchemist/fininfo/README.md) · [JAX kernels](https://github.com/SaadQacif/dynamax/blob/Saad-The-Financial-Markets-Alchemist/fininfo/core.py) · [tests](https://github.com/SaadQacif/dynamax/tree/Saad-The-Financial-Markets-Alchemist/fininfo/tests) · [results](https://github.com/SaadQacif/dynamax/blob/Saad-The-Financial-Markets-Alchemist/fininfo/results/results.json)

## Selected engineering and applied research

### Dynamic modelling for a sulfuric-acid digital twin · team project

In the [ITX H₂SO₄ digital twin](https://github.com/Ismailea4/ITX-H2SO4-Digital-Twin), I contributed a time-aware DCDA process simulator, kinetic-parameter modelling, and forecasting experiments. The dynamic model combines equilibrium and energy-balance calculations, catalyst activity, thermal lag, and sensor noise to generate multivariate sequences for model comparison. In the committed synthetic-data benchmark, the LNN sequence model reports **R² 0.9851** and **MAPE 0.46%**; these are simulator-benchmark results, not live-plant accuracy.

[Dynamic-model report](https://github.com/Ismailea4/ITX-H2SO4-Digital-Twin/blob/main/docs/dynamic_models.md) · [implementation](https://github.com/Ismailea4/ITX-H2SO4-Digital-Twin/tree/main/src/role3_dynamic_models) · [comparison results](https://github.com/Ismailea4/ITX-H2SO4-Digital-Twin/blob/main/results/comparison_matrix.csv)

### Ensemble RL trading application · team project

For an [ensemble trading prototype](https://github.com/Ismailea4/Ensemble-RL-Stock-Trader), I authored a Dockerised application layer with a Streamlit dashboard, FastAPI endpoints, visualisations, and model-loading/backtesting interfaces around PPO, A2C, and DDPG agents. This demonstrates research-software and deployment work; it is not a claim of a profitable live strategy. [My contribution](https://github.com/Ismailea4/Ensemble-RL-Stock-Trader/commit/9a8112e346417c846afd952a164009608b837d13).

### Other explorations

- [Time-series modelling for maritime operations](https://github.com/SaadQacif/Time-Series-for-Maritime-Operations) — classical forecasting and volatility-model implementations with diagnostic notebooks.
- [3D tree-species classification](https://github.com/SaadQacif/Deep-Learning-for-3D-Tree-Species-Classification) — point-cloud sampling and classification experiments.
- [Federated-learning benchmarking](https://github.com/SaadQacif/Federated-Learning-Benchmarking) — algorithm implementations and non-IID data-partitioning experiments.
- [ROS 2 hand-gesture recognition](https://github.com/SaadQacif/ROS2-Hand-Gesture-Recognition) — gesture-model integration with ROS 2 control nodes.

## Education

**MITx · Statistics and Data Science MicroMasters credential** · 2024–2026

Ranked first in cohort; GPA 4.9/5.0. [Credential](https://credentials.edx.org/credentials/3df8955447f449c483424a0e146d8344/) · [letter of accomplishment](https://micromasters.mit.edu/letter/program/5bd0be37-f8e6-4bbf-b723-cf1b6877e9b5). Coursework: probability, statistics, machine learning, and data analysis.

**ENSIAS, Mohammed V University · AI Engineering (2IA)** · 2023–2026

Ranked first in the 2IA promotion, with highest distinction; advanced machine learning, reinforcement learning, and numerical methods.

**Ibn Timiya Preparatory Classes · MPSI / MP\*** · 2021–2023

Mathematics and physics preparation spanning analysis, algebra, topology, differential equations, and probability.

## Tools

Python · C++ · JAX · PyTorch · NumPy · SciPy · pandas · scikit-learn · SQL

## Contact

[qacifsaad@gmail.com](mailto:qacifsaad@gmail.com)
