# Saâd Qacif

**AI engineer and theoretical/applied mathematician, working toward quantitative research.**

Valedictorian of MIT's Statistics & Data Science MicroMasters and in AI
Engineering at ENSIAS with highest distinction. I work where stochastic modelling, statistical
inference, and large-scale computation meet decision-making under uncertainty.

---

### Focus

**Sequential decision-making under uncertainty.** Reinforcement learning with leakage-free
rewards and honest out-of-sample evaluation.

**Stochastic modelling and forecasting.** State-space models, Gaussian processes, Monte-Carlo
simulation, first-principles simulators.

**AI systems engineering.** Multi-agent LLM pipelines with retrieval and tool orchestration,
computer vision for industrial inspection, and models shipped as containerised services with
REST APIs, CI/CD, and C++ inference engines.

**Quantitative finance.** Systematic strategy design, walk-forward backtesting, risk-adjusted
performance.

---

### Projects

**[Law-of-Large-Numbers-Based Graph Deep RL](https://github.com/SaadQacif/Law-of-Large-Numbers-Based-Graph-Deep-RL-Tracking)**
A policy-gradient framework for multi-object state estimation. An input-conditioned
slot-attention policy performs set-to-set completion, graded by optimal (Hungarian) assignment
under a reward that uses no identity labels and explicitly penalises deletion shortcuts.
Sequence-specific biases have non-zero conditional mean but vanish under aggregation, so the
fitted policy estimates the population correction and transfers to held-out sequences.

**[Real-time digital twin, sulfuric-acid DCDA process](https://github.com/Ismailea4/ITX-H2SO4-Digital-Twin)** · *ITX-UM6P / OCP*
Built the dynamic layer: a first-principles stochastic simulator (van 't Hoff equilibrium by
vectorised Newton iteration, adiabatic bed energy balances, catalyst-decay kinetics, sensor
noise) generating 8k multivariate sequences, then benchmarked Liquid Neural Networks, TCNs, and
S4 state-space models on 12-variable horizon forecasting, reaching R² = 0.985 and MAPE 0.46%.
Shipped a static-memory-allocation C++ inference engine.

**[Ensemble RL for systematic trading](https://github.com/Ismailea4/Ensemble-RL-Stock-Trader/commit/9a8112e346417c846afd952a164009608b837d13)**
PPO, A2C, and DDPG agents implemented from scratch over a DOW30 portfolio environment (OHLCV
plus MACD, RSI, CCI, and ADX features) under realistic transaction costs. A rolling-window
validation selector allocates to the agent with the highest out-of-sample Sharpe ratio each
period. Shipped as a Dockerised FastAPI and Streamlit service with equity-curve reporting.

**Stochastic and time-series modelling** · *MIT*
Gaussian-process modelling of ocean-current velocity fields with Monte-Carlo particle
simulation; trend and seasonal decomposition with ARX models and forecast validation via RMSE,
ACF, and PACF diagnostics.

---

### Education

**MIT (MITx)**, MicroMasters in Statistics and Data Science · 2024-2026
Valedictorian, 1st of cohort · GPA 4.9/5.0 ·
[verify](https://credentials.edx.org/credentials/3df8955447f449c483424a0e146d8344/) ·
[letter of accomplishment](https://micromasters.mit.edu/letter/program/5bd0be37-f8e6-4bbf-b723-cf1b6877e9b5)
Probability (6.431x) · Fundamentals of Statistics (18.6501x) · Machine Learning with Python
(6.86x) · Data Analysis (14.310x)

**ENSIAS, Mohammed V University**, AI Engineering (2IA) · 2023-2026
Valedictorian, Pure Mathematics & AI track · Highest Distinction
Ranked 1st of the 2IA promotion. Advanced machine and reinforcement learning algorithms
along with numerical methods.

**Ibn Timiya Preparatory Classes**, MPSI / MP* · 2021-2023
Marrakech. Real and complex analysis, linear algebra, topology, differential equations,
probability, theoretical physics. Admitted to ENSIAS via the national competitive examination.

---

### Toolbox

`Python` `C++` `JAX` `PyTorch` `NumPy` `pandas` `scikit-learn` `SQL`

---

### Contact

[qacifsaad@gmail.com](mailto:qacifsaad@gmail.com)
