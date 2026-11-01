# Saâd Qacif

**AI engineer working across mathematical modelling, statistical inference, and sequential decision-making.**

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=500&size=20&duration=2600&pause=700&color=58A6FF&center=true&vCenter=true&width=900&height=48&lines=Valedictorian+%C2%B7+MITx+Statistics+%26+Data+Science+MicroMasters+%C2%B7+GPA+4.9%2F5;Valedictorian+%C2%B7+AI+Engineering%2C+ENSIAS+%C2%B7+highest+distinction;Model+news+correlation%3A+95%25+intervals+cover+95.0%25%2C+not+91.2%25;Model+source+reliability%3A+coverage+95.1%25%2C+not+72.5%25+under+blind+trust;Right+variance%2C+wrong+shape%3A+a+Gaussian+fit+under-covers+at+z+%3D+%E2%88%923.0;94+tests+cross-check+the+theory+by+Monte+Carlo%2C+quadrature%2C+algebra;Graph-attention+transfer+to+a+4.2%C3%97+sparser+block%3A+recall+23.0%25+%E2%86%92+28.0%25;Simulated+acid-plant+twin%3A+98.5%25+of+variance+explained%2C+0.46%25+error" alt="Valedictorian of the MITx Statistics and Data Science MicroMasters (GPA 4.9/5) and of AI Engineering at ENSIAS with highest distinction. Modelling the correlation between news sources keeps nominal 95% intervals at 95.0% coverage instead of 91.2%. Modelling source reliability gives 95.1% coverage instead of 72.5% under blind trust. A Gaussian with the right variance but the wrong shape under-covers by 3.0 standard errors. 94 tests cross-check the theory by Monte Carlo simulation, numerical quadrature and exact algebra. On a separate block of consecutive frames with about 4.2 times fewer detections, the graph-attention policy raised recall from 23.0% to 28.0%. A simulated acid-plant digital twin explains 98.5% of forecast variance with 0.46% mean absolute percentage error." />

![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
![C++](https://img.shields.io/badge/c%2B%2B-%2300599C.svg?style=for-the-badge&logo=cplusplus&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-%23A142F4.svg?style=for-the-badge&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-%234479A1.svg?style=for-the-badge&logo=postgresql&logoColor=white)

I build research-driven AI systems for decisions under uncertainty. My interests span graph reinforcement learning, stochastic modelling, scientific machine learning, and quantitative research. I was valedictorian of both the MITx Statistics and Data Science MicroMasters and AI Engineering at ENSIAS.

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

The kernels are tested against independent routes: joint-Gaussian Schur complements, SciPy densities, high-order quadrature or Monte Carlo simulation. Integration tests run against real `LinearGaussianSSM.filter` outputs.

In the synthetic misspecification studies, the correctly specified models give 95% intervals that cover 94.97% (correlated packets) and 95.13% (unreliable source). Ignoring the packet correlation drops coverage to 91.15%, and always trusting an unreliable source drops it to 72.54%. These results check the method against its own assumptions; they are not evidence of market predictability.

**Inspect the work:** [method and theorems](https://github.com/SaadQacif/dynamax/blob/Saad-The-Financial-Markets-Alchemist/fininfo/README.md) · [JAX kernels](https://github.com/SaadQacif/dynamax/blob/Saad-The-Financial-Markets-Alchemist/fininfo/core.py) · [tests](https://github.com/SaadQacif/dynamax/tree/Saad-The-Financial-Markets-Alchemist/fininfo/tests) · [results](https://github.com/SaadQacif/dynamax/blob/Saad-The-Financial-Markets-Alchemist/fininfo/results/results.json)

## Featured research · Graph RL for soccer-player tracking

**[Law-of-Large-Numbers-Based Graph Deep RL](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking)** asks how to recover the soccer players a detector misses in noisy frame-by-frame footage. The aim is to recover missing players, not to improve a tracking metric by deleting uncertain detections.

The method has three distinct stages. In the first, an upstream tracking pipeline uses Hungarian-based association to turn raw detections into estimated player positions $\tilde X_t$. In the second, an input-conditioned slot-attention policy $\pi_\theta$, trained by policy gradient, takes those estimated positions and proposes body completions $\hat X_t$ for the players the pipeline has missed. In the third, which happens during training only, Opta positions grade the proposals through nearest-distance scoring. The Opta positions are never inputs to the policy.

Keeping the reference positions out of the inputs means the policy must complete the scene from what the tracker estimated. At inference time it needs nothing beyond the tracker's output.

The law-of-large-numbers argument is the design motivation for training across many stretches of footage rather than one. Suppose the correction that chunk $k$ calls for is a shared correction $g$ plus a chunk-specific bias, $g_k = g + b_k$. Such a bias persists within a chunk because its cause, such as a camera angle or a recurring detector failure, persists. If the biases averaged out over an idealised population of independent chunks, with $\mathbb E[b_k] = 0$, the pooled target would recover the shared correction:

```math
\frac{1}{K} \sum_{k=1}^{K} g_k \;\longrightarrow\; g \qquad \text{almost surely as } K \to \infty .
```

This is a heuristic, not a guarantee. Consecutive chunks of footage are neither independent nor identically distributed, and the argument says nothing about how many chunks are enough. Whether pooling helps is therefore an empirical question, and the experiments below test it on one separate block of frames.

**Results.** The footage is split into chunks of roughly 500 consecutive frames. Recall is the fraction of players the system recovers.

**The key result is that the policy improved recall on a separate block of consecutive frames (chunks 51–100), a block with only about a quarter as many detections.**

| Block | Detected bodies | Threshold | Baseline recall | RL recall | Change |
| --- | ---: | :---: | ---: | ---: | ---: |
| Chunks 0–50 | 205,583 | — | 45.1% | 59.5% (best run) | +14.4 pts |
| Chunks 51–100, transfer test | 48,548 | 0.10 | 23.0% | 28.0% | +5.0 pts |
| Chunks 51–100, transfer test | 48,548 | 0.17 | 23.0% | 27.7% | +4.7 pts |

The transfer gain is smaller than the +14.4 points on chunks 0–50. It is still a relative improvement of about 22% in recall, and it holds at both thresholds. The transfer block contains about 4.2 times fewer detected bodies, so the policy has much less observed context to complete from. That is a plausible reason for the smaller gain, but these experiments do not isolate it as the cause.

The improvement is not free. At threshold 0.10, median localization error rose from 4.37 m to 4.61 m: the policy recovers more players, but places them slightly less precisely. Over the same comparison, the logged "Bodies" metric rose from 5.8 to 8.1.

This shows transfer to a separate block of consecutive frames; it is not evidence of transfer across matches.

The application is soccer, not markets, but the methods carry over. The headline metric is recall, which discourages deleting uncertain predictions; this guards against the tracking analogue of a backtest that improves by quietly dropping losing trades. Reference data that grades the policy never enters its inputs, the same separation of decision-time information from scoring information that helps guard against look-ahead bias. Separating a shared effect from persistent regime-specific bias is a central difficulty of learning across heterogeneous market regimes. Hungarian association is an instance of the optimal-assignment problems behind many matching and allocation tasks.

**Inspect the work:** [training implementation](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/blob/main/scripts/train_graph_rl_body_completion.py) · [matching code](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/matching) · [tests](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/tests) · [experiment notes](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/sessions)

These aggregate results come from private experiment logs; the underlying data, full logs, and trained weights are not public.

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

Valedictorian; GPA 4.9/5.0. [Credential](https://credentials.edx.org/credentials/3df8955447f449c483424a0e146d8344/) · [letter of accomplishment](https://micromasters.mit.edu/letter/program/5bd0be37-f8e6-4bbf-b723-cf1b6877e9b5). Coursework: probability, statistics, machine learning, and data analysis.

**ENSIAS, Mohammed V University · AI Engineering (2IA)** · 2023–2026

Valedictorian of the 2IA promotion, with highest distinction; advanced machine learning, reinforcement learning, and numerical methods.

**Ibn Timiya Preparatory Classes · MPSI / MP\*** · 2021–2023

Mathematics and physics preparation spanning analysis, algebra, topology, differential equations, and probability.

## Contact

[qacifsaad@gmail.com](mailto:qacifsaad@gmail.com)
