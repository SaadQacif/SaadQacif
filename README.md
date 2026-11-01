# Saâd Qacif

**Advanced AI engineer working across graph reinforcement learning, neural-network optimization, probabilistic modelling, and statistical inference.**

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
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-%23000000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Django](https://img.shields.io/badge/Django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%2347A248.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-%230086C9.svg?style=for-the-badge&logo=neo4j&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-%23009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)

I build advanced, research-driven AI systems for decisions under uncertainty: graph-attention policies trained with reinforcement learning, bio-inspired metaheuristics that search neural-network architectures and training settings, and probabilistic state-space models. My interests span graph reinforcement learning, stochastic modelling, scientific machine learning, and quantitative research. I was valedictorian of both the MITx Statistics and Data Science MicroMasters and AI Engineering at ENSIAS.

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

**[Law-of-Large-Numbers-Based Graph Deep RL](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking)** investigates **set-valued state reconstruction under partial observability**: recovering soccer players missed by a noisy video-tracking pipeline, while explicitly measuring the tension between coverage and localization accuracy. It is a tracking experiment, not a claim of financial-market predictability.

**From imperfect observations to a learned set-valued policy.** Let $O_t=\{(\tilde p_{ti},u_{ti})\}_{i=1}^{n_t}$ denote the upstream tracker's estimated positions and match-agnostic features, where $\tilde p_{ti}\in\mathbb R^2$ is pitch-normalized and $u_{ti}$ contains speed, confidence, tracking state, team, and reconstruction metadata. Upstream Hungarian-based association constructs those track estimates; the learned policy does **not** observe the reference Opta set $G_t$ at inference time. Its task is to produce a reconstructed player set $\widehat X_t=F_\theta(O_t)$.

**Attention as a data-dependent interaction graph.** The implementation encodes each observed body as $h_i=\phi_\theta([u_{ti},\tilde p_{ti}])$ and initializes **22 input-conditioned latent slots**. At each of three refinement iterations, the soft links from candidate slot $j$ to observed body $i$ are

```math
\alpha_{ji}^{(\ell)}=
\frac{\exp\!\left(\langle W_Qz_j^{(\ell)},W_Kh_i\rangle/\sqrt d\right)}
{\sum_{r=1}^{n_t}\exp\!\left(\langle W_Qz_j^{(\ell)},W_Kh_r\rangle/\sqrt d\right)},
\qquad m_j^{(\ell)}=\sum_{i=1}^{n_t}\alpha_{ji}^{(\ell)}W_Vh_i.
```

The row-stochastic matrix $A_t^{(\ell)}=[\alpha_{ji}^{(\ell)}]$ is an **input-adaptive weighted bipartite graph** between latent player slots and observations, not a fixed physical adjacency matrix. Residual, feed-forward and layer-normalization operations refine the slot embeddings. The output layer predicts bounded pitch-normalized coordinates and activation decisions:

```math
\widehat p_{tj}=\tanh\!\left(p_{tj}^{(0)}+0.3\tanh f_{\theta,p}(z_{tj}^{(3)})\right),\qquad
\widehat X_t=\left\{D\widehat p_{tj}:\ell_{\theta,j}(O_t)>0\right\},\quad
D=\mathrm{diag}(52.5,34).
```

Here $p_{tj}^{(0)}$ is an attention-weighted initialization from observed positions, $\ell_{\theta,j}$ is an activation logit, and $D$ converts normalized coordinates to metres. The set representation avoids assigning an artificial, permanent identity to each output slot.

**A two-sided statistical scoring problem.** With reference positions $G_t$ used **only to score training and evaluation**, the implemented coverage metric is a directed nearest-neighbour recall at a 6 m tolerance, while localization is measured from active predictions toward reference positions:

```math
\mathrm{Rec}_6(\widehat X_t,G_t)
=\frac{1}{|G_t|}\sum_{g\in G_t}
\mathbf1\!\left\{\min_{x\in\widehat X_t}\|x-g\|_2\le 6\,\mathrm m\right\},
\qquad
E_t=\mathrm{median}_{x\in\widehat X_t}
\min_{g\in G_t}\|x-g\|_2.
```

For each training batch $B$, the script computes $\overline{\mathrm{Rec}}_{6,B}$ as the mean frame recall and $E_B$ as the median distance pooled over active predictions. The implemented, dimensionless reward is

```math
R_B=\min\!\left(\frac{\overline{\mathrm{Rec}}_{6,B}}{0.60},1\right)
+\min\!\left(\frac{3.1\,\mathrm m}{\max(E_B,0.01\,\mathrm m)},1\right).
```

Thus coverage and position error enter as **separate saturating objectives**, making the precision–recall trade-off explicit. These are directed nearest-neighbour metrics, **not** a one-to-one Hungarian matching objective; Hungarian association belongs to the upstream tracking stage.

**Stochastic decisions and the actual optimization surrogate.** During training, Gaussian perturbations of the slot logits induce binary activation actions:

```math
\varepsilon_{tj}\sim\mathcal N(0,0.5^2),\qquad
A_{tj}=\mathbf1\{\ell_{\theta,j}(O_t)+\varepsilon_{tj}>0\},\qquad
\mathbb P_\theta(A_{tj}=1\mid O_t)=\Phi\!\left(\frac{\ell_{\theta,j}(O_t)}{0.5}\right).
```

The script then applies a **score-function-style surrogate**, with advantage $\widehat{\mathcal A}=R_B(A)-R_B(A^{\mathrm{det}})+0.05$ and an average binary log-likelihood evaluated at the noise-perturbed logits:

```math
\mathcal L_{\mathrm{impl}}(\theta)
=-\widehat{\mathcal A}\;\frac{1}{|B|\,22}
\sum_{t\in B}\sum_{j=1}^{22}
\left[A_{tj}\log\sigma(\ell_{tj}+\varepsilon_{tj})
+(1-A_{tj})\log\bigl(1-\sigma(\ell_{tj}+\varepsilon_{tj})\bigr)\right].
```

This formula describes the **implemented update**, rather than claiming an exact policy-gradient theorem: thresholded Gaussian sampling has a probit action distribution, while the surrogate uses a sigmoid likelihood. Moreover, the position head receives no direct gradient through this discrete reward computation. These are concrete optimization limitations, not established convergence or optimality guarantees.

**Law of large numbers: the assumption and its failure mode.** The central statistical intuition is that the correction needed for video chunk $k$ can be decomposed into a shared component $g$ and a chunk-specific distortion $b_k$, caused for example by camera viewpoint or recurring detection errors. **If** $b_k$ are integrable, independent and identically distributed with zero mean, the strong law yields

```math
g_k=g+b_k,\qquad \mathbb E[b_k]=0,\qquad
\widehat g_K=\frac{1}{K}\sum_{k=1}^{K}g_k
\longrightarrow g
\quad (K\to\infty,\ \mathrm{a.s.})
```

That limiting result is **conditional**, not an empirical theorem about consecutive football footage. In the non-independent case, the uncertainty of the pooled bias instead depends on all cross-chunk covariances:

```math
\mathrm{Cov}(\overline b_K)
=\frac{1}{K^2}\sum_{k=1}^{K}\sum_{l=1}^{K}
\mathrm{Cov}(b_k,b_l),\qquad
\overline b_K=\frac{1}{K}\sum_{k=1}^{K}b_k.
```

Persistent temporal dependence can therefore make $K$ chunks worth substantially fewer than $K$ independent observations. Neither zero-mean bias nor the required mixing conditions have been demonstrated for these experiments; **cross-chunk generalization remains an empirical claim**.

**Results.** Footage was divided into chunks of roughly 500 consecutive frames. The reported recall measures how many reference players are recovered within the localization tolerance.

**The key empirical result is improved recall on a separate, substantially sparser block of consecutive frames (chunks 51–100).**

| Block | Detected bodies | Threshold | Baseline recall | RL recall | Change |
| --- | ---: | :---: | ---: | ---: | ---: |
| Chunks 0–50 | 205,583 | — | 45.1% | 59.5% (best run) | +14.4 pts |
| Chunks 51–100, transfer test | 48,548 | 0.10 | 23.0% | 28.0% | +5.0 pts |
| Chunks 51–100, transfer test | 48,548 | 0.17 | 23.0% | 27.7% | +4.7 pts |

On the transfer block, the absolute gain of **5.0 percentage points** corresponds to approximately **21.7% relative recall improvement** at threshold 0.10; at threshold 0.17, the gain is 4.7 points. The transfer block contains approximately **4.2 times fewer detected bodies** than chunks 0–50. This change in observational density is a plausible contributor to the smaller gain, but its causal role has not been isolated.

The gain carries a measurable cost: at threshold 0.10, the median localization error rose from **4.37 m to 4.61 m**, while the logged **Bodies** metric rose from **5.8 to 8.1**. The evidence supports **transfer between two blocks of consecutive frames**, not across distinct matches.

**Quantitative-research relevance.** The transferable ideas are methodological, not domain-specific: (i) a policy must act only on information available at decision time, (ii) coverage and error should be assessed jointly to avoid optimizing a metric by discarding difficult cases, and (iii) dependence across regimes directly limits how much one may infer from pooled samples. These are relevant to portfolio selection, market-regime generalization, and leakage-controlled backtesting, without implying that the tracking model is itself a trading strategy.

**Inspect the work:** [training implementation](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/blob/main/scripts/train_graph_rl_body_completion.py) · [matching code](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/matching) · [tests](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/tests) · [experiment notes](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking/tree/main/sessions)

These aggregate results come from private experiment logs; the underlying data, full logs, and trained weights are not public.



## Selected engineering and applied research

### [Artificial Bee Colony optimization library · released Python package](https://github.com/SaadQacif/Bio-inspired-Project-Meta-heuristics-Optimization-)

[![Tests](https://github.com/SaadQacif/Bio-inspired-Project-Meta-heuristics-Optimization-/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/SaadQacif/Bio-inspired-Project-Meta-heuristics-Optimization-/actions/workflows/tests.yml)
[![GitHub release](https://img.shields.io/github/v/release/SaadQacif/Bio-inspired-Project-Meta-heuristics-Optimization-?logo=github&label=release)](https://github.com/SaadQacif/Bio-inspired-Project-Meta-heuristics-Optimization-/releases/latest)

Turned swarm-intelligence research code into an installable, tested **Python library**, `abc_framework`. Six Artificial Bee Colony variants (standard ABC, simulated-annealing hybrids, a Q-learning-controlled RL-ABC, Queen ABC, multi-colony Queen ABC, and an ABC–flower-pollination hybrid) share one contract: an exact objective-evaluation budget, the best-ever solution returned, seeded reproducibility, and validated settings. The package also tunes SVM hyperparameters with feature scaling fitted inside each cross-validation fold, so no validation or test data leaks into preprocessing.

An **equal-budget benchmark** (20,000 evaluations, 15 paired seeds, Wilcoxon tests with Holm correction) shows that no variant significantly outperforms standard ABC on six classical functions; the saved runs and summaries ship with the release. GitHub Actions tests Python 3.10–3.14 and verifies the built wheel in a clean environment; the release files were built from the tagged commit, tested before upload, and published with SHA-256 checksums.

```bash
pip install https://github.com/SaadQacif/Bio-inspired-Project-Meta-heuristics-Optimization-/releases/download/v0.1.0/abc_framework-0.1.0-py3-none-any.whl
```

[v0.1.0 release](https://github.com/SaadQacif/Bio-inspired-Project-Meta-heuristics-Optimization-/releases/tag/v0.1.0) · [benchmark results](https://github.com/SaadQacif/Bio-inspired-Project-Meta-heuristics-Optimization-/tree/main/benchmarks) · [PolyForm Noncommercial license](https://github.com/SaadQacif/Bio-inspired-Project-Meta-heuristics-Optimization-/blob/main/LICENSE.md)

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
