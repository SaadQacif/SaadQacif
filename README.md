# Saâd Qacif

**AI engineer working across mathematical modelling, statistical inference, and sequential decision-making.**

I build research-driven AI systems for decisions under uncertainty. My interests span graph reinforcement learning, stochastic modelling, scientific machine learning, and quantitative research. I was ranked first in my MITx Statistics and Data Science MicroMasters cohort and in AI Engineering at ENSIAS.

## Featured research · Decision-aware financial information transport

**[fininfo](https://github.com/SaadQacif/dynamax/tree/Saad-The-Financial-Markets-Alchemist/fininfo)** is an extension of [Dynamax](https://github.com/probml/dynamax) that implements the draft *Decision-Aware Financial Information Transport in State-Space Models*. It asks what a piece of news or data is worth to a financial decision. Instead of a similarity score, it uses a likelihood: the packet is a linear-Gaussian experiment on a latent market state,

```math
Z \sim \mathcal N(m, P), \qquad S = MZ + \varepsilon, \qquad \varepsilon \sim \mathcal N(0, R),
```

and the posterior update is carried through the state dynamics to a payoff $Y_h = B_h Z + d_h + \eta_h$. With $\Sigma = MPM^\top + R$ and $K = PM^\top\Sigma^{-1}$, the realised revision of the payoff forecast, its ex-ante covariance and the information it carries are

```math
\delta_h(s) = B_h K (s - Mm), \qquad \Delta_h = B_h P M^\top \Sigma^{-1} M P B_h^\top, \qquad I(Y_h; S) = \frac{1}{2} \log \frac{\det \Sigma_h}{\det \Sigma_{h,+}} \le I(Z; S).
```

These quantities are invariant under any invertible change of state or signal coordinates, so the financial conclusions do not depend on how the latent model is parameterised. Large state information is also no guarantee of financial relevance: a packet can be arbitrarily precise about the state while leaving every payoff-relevant canonical mode untouched.

The economic layer expresses information in currency. Take positions in a closed convex set with a fixed quadratic cost $H \succ 0$. The expected gain from observing the packet then satisfies $0 \le \mathbb E f(\mu_S) - f(\mu) \le \tfrac12 \mathrm{tr}(H^{-1}\Delta)$, with equality when positions are unrestricted. For an investor with exponential utility and risk aversion $\gamma$, the exact ex-ante fee that makes buying the signal a matter of indifference is

```math
c_{\text{info}} = \frac{1}{2\gamma} \log \frac{\det \Sigma}{\det \Sigma_+} = \frac{I(Y; S)}{\gamma}.
```

The package also covers the cases where this simple picture breaks down. Correlated packets are residualised with $A = R_{21}R_{11}^{-1}$, so sequential updating reproduces the batch update exactly. Shapley attribution of packets is nonnegative and efficient. A two-packet counterexample with $v(\lbrace 2\rbrace) = 0 \lt v(\lbrace 1,2\rbrace) - v(\lbrace 1\rbrace)$ nonetheless shows that the value of information is not submodular, so greedy selection guarantees do not carry over. Integrated-gradient attribution over transition edges sums exactly to the change in the response. Yet it depends on the chosen basis, so individual latent links have no intrinsic economic meaning. Discounted cash flows are projected with their cross-horizon dependence intact. Finally, a source that may be uninformative, with an arbitrary null density $f_0$, yields a posterior mixture whose variance can exceed the prior's.

Each kernel is tested against an independent route: joint-Gaussian Schur complements, SciPy densities, high-order quadrature or Monte Carlo simulation. Integration tests run against real `LinearGaussianSSM.filter` outputs.

In the synthetic misspecification studies, the correctly specified models give 95% intervals that cover 94.97% (correlated packets) and 95.13% (unreliable source). Ignoring the packet correlation drops coverage to 91.15%, and always trusting an unreliable source drops it to 72.54%. These results check the method against its own assumptions; they are not evidence of market predictability.

**Inspect the work:** [method and theorems](https://github.com/SaadQacif/dynamax/blob/Saad-The-Financial-Markets-Alchemist/fininfo/README.md) · [JAX kernels](https://github.com/SaadQacif/dynamax/blob/Saad-The-Financial-Markets-Alchemist/fininfo/core.py) · [tests](https://github.com/SaadQacif/dynamax/tree/Saad-The-Financial-Markets-Alchemist/fininfo/tests) · [results](https://github.com/SaadQacif/dynamax/blob/Saad-The-Financial-Markets-Alchemist/fininfo/results/results.json)

## Featured research · Graph RL for soccer-player tracking

**[Law-of-Large-Numbers-Based Graph Deep RL](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking)** asks how to recover the soccer players a detector misses in noisy frame-by-frame footage. It deliberately rules out the cheap route of improving a tracking metric by deleting uncertain detections.

At each frame the detector returns an incomplete and noisy set of positions, and the task is to output the full set of players on the pitch. The project treats this as set-to-set completion. An input-conditioned slot-attention policy $\pi_\theta$ proposes a position and an activity decision for each slot, and it is trained by policy gradient. Because the output is a set, the policy is scored through an optimal one-to-one (Hungarian) assignment between predicted players $\hat x_i$ and reference players $x_j$,

```math
\sigma^\star = \arg\min_{\sigma} \sum_{i} \lVert \hat x_i - x_{\sigma(i)} \rVert ,
```

so the feedback needs no identity labels and does not depend on the order of the slots. The reward combines set-level recall with localization error over the matched pairs, and it explicitly penalises deletion, so dropping an uncertain detection cannot buy a better score.

The law-of-large-numbers argument concerns what such a policy actually learns. Within one sequence $k$, the correction the data calls for is a population correction $g$ plus a sequence-specific bias, $g_k = g + b_k$. That bias has a nonzero mean conditional on the sequence, because whatever causes it, such as a camera angle or a recurring detector failure, persists throughout that sequence. Across sequences drawn independently from the same population, however, $\mathbb E[b_k] = 0$, and so

```math
\frac{1}{K} \sum_{k=1}^{K} g_k \;\longrightarrow\; g \qquad \text{almost surely as } K \to \infty .
```

A policy fitted on many varied sequences therefore targets the population correction $g$ rather than the idiosyncrasies of any single sequence, in the same way that a pooled estimator averages out regime-specific error. Whether this transfer holds on held-out sequences is an empirical question that the public extract does not settle.

The application is soccer, not markets, but the reasoning carries over directly. A score that improves when uncertain predictions are discarded is the tracking analogue of a backtest that improves by quietly dropping losing trades; the remedy in both cases is an objective under which the shortcut cannot pay. Separating a population-level effect from persistent regime-specific bias is the central difficulty of learning from heterogeneous market regimes. Set-valued prediction under optimal assignment is the structure behind many matching and allocation problems.

**Inspect the work:** [training implementation](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/blob/main/scripts/train_graph_rl_body_completion.py) · [matching code](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/matching) · [tests](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/tests) · [experiment notes](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/sessions)

The public repository is a code-only research extract; licensed media, source data, generated outputs, and trained weights are not redistributed. Cross-sequence transfer remains a research question rather than a result established by this public extract.

## Selected engineering and applied research

### Dynamic modelling for a sulfuric-acid digital twin · team project

In the [ITX H₂SO₄ digital twin](https://github.com/Ismailea4/ITX-H2SO4-Digital-Twin), I contributed a time-aware DCDA process simulator, kinetic-parameter modelling, and forecasting experiments. The dynamic model combines equilibrium and energy-balance calculations, catalyst activity, thermal lag, and sensor noise to generate multivariate sequences for model comparison. In the committed synthetic-data benchmark, the LNN sequence model reports **R² 0.9851** and **MAPE 0.46%**; these are simulator-benchmark results, not live-plant accuracy.

[Dynamic-model report](https://github.com/Ismailea4/ITX-H2SO4-Digital-Twin/blob/main/docs/dynamic_models.md) · [implementation](https://github.com/Ismailea4/ITX-H2SO4-Digital-Twin/tree/main/src/role3_dynamic_models) · [comparison results](https://github.com/Ismailea4/ITX-H2SO4-Digital-Twin/blob/main/results/comparison_matrix.csv)

### Ensemble RL trading application · team project

For an [ensemble trading prototype](https://github.com/Ismailea4/Ensemble-RL-Stock-Trader), I authored a Dockerised application layer with a Streamlit dashboard, FastAPI endpoints, visualisations, and model-loading/backtesting interfaces around PPO, A2C, and DDPG agents. This demonstrates research-software and deployment work; it is not a claim of a profitable live strategy. [My contribution](https://github.com/Ismailea4/Ensemble-RL-Stock-Trader/commit/9a8112e346417c846afd952a164009608b837d13).

### Other explorations

Earlier projects include [classical forecasting and volatility models for maritime operations](https://github.com/SaadQacif/Time-Series-for-Maritime-Operations) with diagnostic notebooks, [point-cloud sampling and classification of 3D tree species](https://github.com/SaadQacif/Deep-Learning-for-3D-Tree-Species-Classification), [benchmarking of federated-learning algorithms under non-IID data partitions](https://github.com/SaadQacif/Federated-Learning-Benchmarking), and the [integration of a hand-gesture recognition model with ROS 2 control nodes](https://github.com/SaadQacif/ROS2-Hand-Gesture-Recognition).

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
