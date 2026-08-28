SUMMARY: Chapter-by-chapter ideal coverage spec for a conventional-ML interview-prep volume (2026 loops at frontier labs and platform companies): 8 content chapters (supervised core; trees/ensembles; unsupervised; probabilistic foundations; applied craft; experimentation/causal; time series; from-scratch coding implementables) plus a diff checklist to run against the user's own notes. Each chapter lists must-know subtopics, whiteboard derivations, representative interview questions with 2-line sketches, red-flag wrong answers, and page-weight guidance. Overlap with the deep-learning volume was checked against sections/02_learning_theory.tex (bias-variance identity + red flags already covered), 18_evaluation_metrics.tex (classification/ranking metrics, calibration methods, significance testing, power analysis, offline-online gap), 17_production_systems.tex (A/B basics, peeking, interleaving/bandits/switchback table, drift/PSI), and 19_decision_frameworks.tex (a strong GBM-vs-NN-for-tabular question already exists) — the spec marks where the new volume should cross-reference rather than duplicate.

# Ch. 1 — Supervised Learning Core: Linear Models, SVMs, Naive Bayes, kNN
**Target page weight: 14–18% of the volume (~25–32 pp).** This chapter is the derivation backbone: interviewers use linear/logistic regression as the canonical 'can you do math on a whiteboard' probe, and the regularization-geometry question is arguably the single most-asked classical question after trees.

### 1.1 Linear regression
Must-know:
- Model + assumptions (linearity, independence, homoscedasticity, exogeneity; normality only needed for inference, not for OLS to be BLUE — Gauss-Markov).
- **Closed form vs gradient descent**: derive the normal equations $\hat\beta = (X^TX)^{-1}X^Ty$ from $\nabla_\beta \|y-X\beta\|^2 = 0$; complexity $O(nd^2 + d^3)$ vs GD's $O(ndk)$; when closed form fails ($d$ large, $X^TX$ singular/ill-conditioned, streaming data); the SVD/pseudoinverse fix and why libraries use QR/Cholesky, not explicit inversion.
- Geometric view: OLS as orthogonal projection of $y$ onto col($X$); hat matrix, leverage.
- Multicollinearity: symptoms (unstable coefficients, huge variance), diagnostics (VIF, condition number), remedies (ridge, drop/combine features).

### 1.2 Regularization (L1/L2/elastic net)
Must-know:
- Ridge closed form $(X^TX + \lambda I)^{-1}X^Ty$; why ridge always has a unique solution even when $n < d$.
- **The geometry answer, done right**: constrained-form equivalence (penalty ↔ constraint via Lagrangian), L1 ball has corners on axes → solutions hit corners → exact zeros; L2 ball is rotationally smooth → shrinkage without sparsity. Should include the contour-plot picture AND the soft-thresholding derivation for the orthonormal case: $\hat\beta_j = \text{sign}(\beta_j^{OLS})(|\beta_j^{OLS}| - \lambda)_+$ vs ridge's uniform scaling $\beta^{OLS}/(1+\lambda)$.
- Bayesian view: L2 = Gaussian prior MAP, L1 = Laplace prior MAP (cross-ref Ch. 4).
- Practicalities: standardize features first, don't penalize intercept, elastic net for correlated feature groups (L1 alone picks one arbitrarily).

### 1.3 Logistic regression and the GLM view
Must-know:
- Derive from Bernoulli likelihood: log-odds linear model, sigmoid as inverse link; NLL = cross-entropy; show gradient $\nabla = X^T(\sigma(X\beta) - y)$ — the famously clean form, and note it's identical in shape to the linear-regression gradient (GLM unification).
- Why no closed form; Newton/IRLS as the classical fit method; convexity of the NLL.
- Complete/quasi-complete separation → weights diverge → regularization as the fix (a favorite probe).
- **GLM framing**: exponential family, link functions, canonical links; linear/logistic/Poisson/softmax as one family; why canonical link makes the gradient take the (prediction − target) form.
- Multiclass: softmax regression, one-vs-rest, and the overparameterization/identifiability note.
- Interpretation: coefficients as log-odds ratios; why comparing raw coefficient magnitudes without standardization is wrong.

### 1.4 SVM and kernels
Must-know:
- **Margin derivation**: geometric margin $= 1/\|w\|$ under canonical scaling, primal hard-margin QP; soft margin with slack + hinge-loss reformulation $\min \|w\|^2/2 + C\sum \max(0, 1-y_i f(x_i))$ — candidates must be able to move between constrained and loss-function forms.
- **Dual derivation**: Lagrangian, KKT conditions, dual QP in $\alpha$, why data appears only through inner products → kernel trick; support vectors as the points with $\alpha_i > 0$; sparsity of the solution.
- Kernels: definition (Mercer/PSD condition), RBF as implicit infinite-dimensional feature map, RBF hyperparameters ($\gamma$, $C$) and their bias-variance roles; kernel matrix cost $O(n^2)$ as the scaling killer.
- Hinge vs logistic loss comparison plot; SVM gives no probabilities (Platt scaling as the patch — cross-ref calibration, Ch. 4/DL vol. ch. 18).
- **Honest 2026 relevance note (must be present)**: SVMs are rarely the production choice (GBMs and NNs won), but they persist in interviews because the dual/kernel derivation tests optimization literacy; kernel ideas survive in Gaussian processes, kernel regression view of attention, and NTK theory. A candidate should say this rather than pretend SVMs are a live default.

### 1.5 Naive Bayes
Must-know: generative vs discriminative framing (NB vs LR as the canonical pair — Ng & Jordan asymptotics: NB converges faster with less data, LR wins asymptotically); conditional-independence assumption and why NB still ranks well when it's violated (miscalibrated but directionally right); Laplace smoothing and why unsmoothed zero counts annihilate the posterior; Gaussian vs multinomial vs Bernoulli variants; log-space computation for underflow.

### 1.6 kNN
Must-know: nonparametric, zero training cost / $O(nd)$ query cost; $k$ as the bias-variance knob (k=1 → zero training error, high variance; k=n → predicts prior); curse of dimensionality — distances concentrate, ratio of nearest-to-farthest → 1, so kNN degrades in high-d (be able to sketch why volume concentrates in the shell); need for feature scaling; distance metric choices; weighted kNN; ANN methods (kd-tree, LSH, HNSW) as one paragraph pointer — kd-tree mechanics live in Ch. 8.

### Whiteboard derivations for this chapter
1. Normal equations from scratch + when to prefer GD.
2. Logistic NLL gradient (the σ(x)−y form).
3. L1 sparsity: geometric picture + soft-thresholding in the orthonormal case.
4. SVM: primal → Lagrangian → dual → kernel trick, and identify support vectors from KKT.
5. NB decision rule from Bayes' rule with the independence assumption.

### Common interview questions (title + sketch)
- **'Why does L1 give sparsity and L2 doesn't?'** — Corners-of-the-ball geometry plus subgradient/soft-thresholding argument; strong answers give both the picture and the math, and mention the Laplace-prior view.
- **'Derive logistic regression's loss from maximum likelihood.'** — Bernoulli likelihood → NLL → cross-entropy; then show the gradient and note convexity, no closed form, IRLS.
- **'When would you use logistic regression over gradient boosting in 2026?'** — Interpretability/regulatory needs, tiny data, well-understood linear signal, extreme-scale online serving with sparse features (ad CTR heritage), calibrated probabilities out of the box; not 'never'.
- **'Your logistic regression weights blew up to ±1e6. What happened?'** — Perfect separation; likelihood is maximized at infinity; fix with L2 or priors.
- **'What does the C parameter in SVM do? What about γ in RBF?'** — C: inverse regularization (large C → hard margin → overfit); γ: kernel width (large γ → wiggly boundary → overfit); both are bias-variance dials.
- **'Generative vs discriminative — define and give trade-offs.'** — Model P(x,y) vs P(y|x); NB/LR pair, sample-efficiency vs asymptotic-accuracy trade, generative handles missing features and can generate data.
- **'Why does kNN fail in high dimensions?'** — Distance concentration; all points nearly equidistant; also cost. Strong answers quantify (volume in shell) and mention learned embeddings as the modern fix.

### Red-flag wrong answers
- 'L1 is better because it's a stronger penalty' (confuses sparsity mechanism with strength).
- Inverting $X^TX$ as the recommended implementation (numerically naive; QR/Cholesky/SVD).
- 'Logistic regression has a closed-form solution.'
- 'OLS requires normally distributed features/targets' (normality is for inference, and it's on errors, not features).
- Claiming the kernel trick 'computes the high-dimensional features efficiently' (it never computes them at all — only inner products).
- 'SVMs maximize accuracy' / can't state what a support vector is.
- Treating NB output probabilities as calibrated.
- Not scaling features before kNN/SVM/regularized regression.

EXISTING COVERAGE: The DL volume touches regularization only from the deep-learning side (weight decay as variance control in sections/02_learning_theory.tex, e.g. the follow-up 'Why does weight decay reduce variance?' at line ~82); no L1/L2 geometry, no GLM, no SVM/kernels, no NB/kNN anywhere. sections/19_decision_frameworks.tex line ~415 ('Start Simple, Add Complexity') names logistic regression as the baseline rung — the new volume can cross-reference that framing instead of restating it. Everything else in this chapter is net-new.

# Ch. 2 — Trees and Ensembles (the #1 classical interview area)
**Target page weight: 18–22% of the volume (~32–40 pp) — the largest chapter.** This is where applied-company loops spend the most classical-ML time, and where breadth rounds at frontier labs check that a candidate's knowledge isn't 'transformers all the way down'.

### 2.1 Decision tree mechanics
Must-know:
- Recursive binary splitting; greedy top-down induction; why globally optimal trees are NP-hard so greedy is what everyone does.
- **Impurity measures, with formulas**: Gini $\sum p_k(1-p_k)$, entropy $-\sum p_k \log p_k$, misclassification error; information gain = parent impurity − weighted child impurity; why Gini/entropy are preferred over misclassification for splitting (strictly concave → sensitive to probability changes even when the argmax class doesn't flip); for regression: variance reduction / SSE.
- Split finding cost: sort each feature once, $O(n \log n \cdot d)$ per node via sorted scan; handling of continuous features (midpoints between sorted values) — this connects directly to the Ch. 8 coding task.
- Stopping/regularization: max depth, min samples per leaf, min impurity decrease; pre- vs post-pruning (cost-complexity pruning with the $\alpha$ path).
- Properties: invariance to monotone feature transforms (no scaling needed), native handling of mixed feature types, axis-aligned boundaries (hence trouble with diagonal decision boundaries and with smooth functions — trees are piecewise-constant), instability (high variance) as the motivation for ensembles, inability to extrapolate beyond the training range (key for time series, cross-ref Ch. 7).

### 2.2 Bagging and Random Forests
Must-know:
- Bootstrap sampling; **the variance-reduction algebra**: averaging $B$ estimators with pairwise correlation $\rho$ gives $\text{Var} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ — derive it; hence bagging attacks variance not bias, and the residual $\rho\sigma^2$ term explains why RF adds feature subsampling (mtry) to decorrelate trees.
- OOB estimation: each tree sees ~63.2% of samples ($1 - 1/e$ — be able to derive the limit $(1-1/n)^n \to e^{-1}$); OOB error as free cross-validation; OOB-based feature importance.
- RF importances: mean decrease in impurity (biased toward high-cardinality/continuous features) vs permutation importance (cross-ref Ch. 5 for the traps).
- Why RF barely overfits with more trees (variance monotonically decreases in B; bias fixed) — but each tree deep/unpruned.

### 2.3 Gradient boosting derived properly
Must-know (this derivation is the chapter's centerpiece — whiteboard-grade):
- Additive modeling / forward stagewise: $F_m = F_{m-1} + \eta h_m$.
- **Functional gradient descent**: at each stage fit $h_m$ to the negative gradient of the loss evaluated at current predictions, $r_i = -\partial L(y_i, F(x_i))/\partial F(x_i)$; show that for squared loss the pseudo-residuals are literal residuals, and for log-loss they are $y_i - p_i$; hence 'boosting = gradient descent in function space, one tree per step'.
- AdaBoost as the historical special case (exponential loss, sample reweighting view) — one page, mainly to answer 'how does AdaBoost relate to gradient boosting?'.
- Shrinkage (learning rate) vs number of trees trade; subsampling (stochastic gradient boosting); why boosting attacks bias (sequential error-fixing) while bagging attacks variance — and why boosting CAN overfit with too many rounds while RF essentially can't.

### 2.4 XGBoost / LightGBM / CatBoost internals
Must-know:
- **Second-order (Newton) objective**: Taylor-expand the loss to second order, derive the optimal leaf weight $w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$ and the split-gain formula $\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$; explain the explicit regularization ($\lambda$ on leaf weights, $\gamma$ per leaf) baked into the objective — this derivation is asked at senior loops and separates 'used XGBoost' from 'understands XGBoost'.
- Histogram-based split finding (bin features to 255 buckets, gradient histograms, histogram subtraction trick) — why LightGBM is fast.
- **Leaf-wise (best-first, LightGBM) vs level-wise (depth-wise, classic XGBoost) growth**: leaf-wise gets lower loss for the same leaf count but overfits small data without depth limits.
- LightGBM extras at name-drop depth: GOSS (keep large-gradient samples, subsample small-gradient ones), EFB (bundle mutually-exclusive sparse features).
- **Categorical handling**: one-hot vs native splits; LightGBM's sorted-by-gradient-statistics k-way splits; CatBoost's ordered target statistics and why 'ordered' — prevents target leakage from naive target encoding (cross-ref Ch. 5 leakage taxonomy); ordered boosting to fight prediction shift.
- Missing-value handling: XGBoost learns a default direction per split.
- Sparsity-aware split finding; approximate split finding with weighted quantile sketch (one paragraph).
- Hyperparameter map with tuning priorities: n_estimators + learning_rate (jointly, with early stopping), max_depth/num_leaves, min_child_weight, subsample/colsample, lambda/alpha — and which combat overfitting vs underfitting.

### 2.5 GBM vs NN for tabular: judgment
Keep this SHORT here (1–2 pp) and cross-reference the DL volume's decision-frameworks chapter, which already contains a full interview question on exactly this. The new volume should add only the classical-side detail the DL volume lacks: why tree ensembles' inductive bias fits tabular data (piecewise-constant, robust to uninformative features and heavy-tailed marginals — Grinsztajn et al.'s three explanations), TabPFN/foundation-model-for-tabular as the 2025–26 update, and the practical ensemble-of-both pattern.

### Whiteboard derivations
1. Information gain computation on a small concrete dataset (numbers, not just formulas).
2. Variance of correlated averages → why RF subsamples features.
3. $(1-1/n)^n \to 1/e$ for OOB.
4. Functional gradient descent: pseudo-residuals for squared loss and log loss.
5. XGBoost leaf weight and gain from the second-order expansion.

### Common interview questions
- **'Explain how gradient boosting works to someone who knows gradient descent.'** — Gradient descent in function space; each tree approximates the negative gradient; learning rate = step size. The 'residual fitting' story is the squared-loss special case, not the definition.
- **'Random forest vs gradient boosting — when each?'** — Variance vs bias reduction, parallel vs sequential, tuning robustness vs peak accuracy, overfitting behavior with ensemble size; RF when you want low-tuning robustness/OOB, GBM when you want SOTA tabular accuracy.
- **'Derive the optimal leaf value in XGBoost.'** — Second-order Taylor, sum g/h per leaf, minimize quadratic → $-G/(H+\lambda)$; then the gain formula.
- **'Why does LightGBM use leaf-wise growth, and what's the risk?'** — Best-first growth maximizes loss reduction per leaf; risk is deep lopsided trees overfitting small datasets; num_leaves + max_depth as the controls.
- **'Why is Gini preferred over misclassification error for splits?'** — Strict concavity: rewards purer children even when predicted class is unchanged; misclassification error is flat over probability regions.
- **'Your GBM's training loss keeps dropping but validation loss rose after round 300. What do you tune?'** — Early stopping first; then learning rate down + more rounds, depth/num_leaves down, min_child_weight up, subsampling; mention lr/rounds joint trade.
- **'How does CatBoost avoid target leakage with categorical features?'** — Ordered target statistics: encode each row using only 'previous' rows in a permutation; plus ordered boosting for prediction shift.
- **'Why can't a tree/GBM extrapolate a trend?'** — Piecewise-constant prediction bounded by training-leaf values; matters for time series with trend (detrend or model trend separately).

### Red-flag wrong answers
- 'Gradient boosting fits residuals' stated as the general mechanism with no awareness it's the squared-loss special case.
- 'Random forests reduce bias' / 'boosting reduces variance' (inverted).
- 'More trees in RF overfits' (confusing with boosting rounds).
- Not knowing OOB or claiming bootstrap samples use ~all points.
- 'XGBoost is just gradient boosting but faster' (misses second-order objective + regularization in the objective).
- Believing trees need normalized features.
- One-hot encoding a 10K-cardinality categorical into a GBM without hesitation (cardinality explosion; use native/target encoding).
- Impurity-based feature importance quoted as ground truth (cardinality bias, correlation splitting).

EXISTING COVERAGE: sections/19_decision_frameworks.tex contains a full L5 interview question 'When should you use gradient boosting vs. neural networks for tabular data?' (lines ~738–762) with the Grinsztajn et al. citation, criteria lists, red flags, and follow-ups (TabNet/FT-Transformer, high-cardinality categoricals, hybrid ensembles); its architecture table (lines ~50–83) also names XGBoost/LightGBM as first choice for tabular. The new volume's §2.5 should be a thin classical-side complement that cross-references that question rather than re-deriving the judgment call. No DL-volume coverage exists of tree mechanics, bagging/RF math, the boosting derivation, or booster internals — all net-new.

# Ch. 3 — Unsupervised Learning: Clustering, EM, PCA, Manifold Methods
**Target page weight: 12–15% of the volume (~22–27 pp).** K-means and PCA are the two most common 'derive it / implement it' unsupervised asks; GMM/EM is the standard depth-probe at frontier labs.

### 3.1 k-means
Must-know:
- Objective: minimize within-cluster SSE $\sum_k \sum_{i \in C_k} \|x_i - \mu_k\|^2$; Lloyd's algorithm = alternating minimization (assignment step optimal given centroids, centroid step optimal given assignments → derive that the mean minimizes SSE by setting gradient to zero); hence monotone objective decrease and finite-step convergence — **to a local optimum only**.
- **k-means++**: seed with D² sampling (probability ∝ squared distance to nearest chosen center); the $O(\log k)$-competitive guarantee at name-drop depth; why random init needs restarts.
- Choosing k: elbow (and its subjectivity), silhouette score, gap statistic; 'there is no ground truth k' as an honest framing.
- **Failure modes (must be enumerable on demand)**: non-spherical clusters (moons/rings), unequal cluster sizes and densities, different variances (k-means implicitly assumes equal isotropic covariance — see GMM connection), sensitivity to outliers (means, not medoids), curse of dimensionality on Euclidean distance, categorical features (k-modes as pointer), need for scaling.
- Connections: k-means = GMM with hard assignments and $\sigma^2 \to 0$ isotropic covariances; minibatch k-means for scale.

### 3.2 GMM and EM (derive both steps)
Must-know:
- Mixture model likelihood; why direct MLE is intractable (log of a sum); latent variable formulation.
- **E-step derivation**: responsibilities $\gamma_{ik} = \frac{\pi_k N(x_i|\mu_k,\Sigma_k)}{\sum_j \pi_j N(x_i|\mu_j,\Sigma_j)}$ via Bayes' rule.
- **M-step derivation**: maximize expected complete-data log-likelihood; derive $\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$, similarly $\Sigma_k$, $\pi_k$ (with the Lagrange multiplier for $\sum \pi_k = 1$).
- **Why EM works**: ELBO / Jensen's-inequality argument — E-step tightens the bound, M-step maximizes it; monotone likelihood ascent; local optima; connection to variational inference as one forward-looking paragraph (bridges to the DL volume's VAE material).
- Practical: covariance constraints (full/diag/spherical/tied) as a bias-variance dial; singularity collapse (a component shrinking onto one point → infinite likelihood) and the regularization fix; model selection via BIC/AIC.
- Soft vs hard assignments; when GMM beats k-means (elliptical clusters, density estimation, calibrated cluster membership).

### 3.3 Hierarchical clustering and DBSCAN
Must-know (survey depth, ~3 pp):
- Agglomerative: linkage criteria (single/complete/average/Ward) and their characteristic failure shapes (single-link chaining, complete-link crowding); dendrograms; $O(n^2 \log n)$–$O(n^3)$ cost; no need to pre-specify k.
- DBSCAN: core/border/noise points, eps + minPts, density-reachability; finds arbitrary shapes, labels outliers natively, no k; fails with varying densities and is sensitive to eps in high-d; HDBSCAN as the modern default name-drop; k-distance plot for eps selection.
- One comparison table: k-means / GMM / hierarchical / DBSCAN across shape assumptions, scalability, outlier handling, hyperparameters.

### 3.4 PCA
Must-know:
- **Variance-maximization derivation**: maximize $w^T S w$ s.t. $\|w\|=1$ → Lagrangian → $Sw = \lambda w$ → top eigenvector; subsequent components via orthogonality constraint.
- Equivalent reconstruction-error minimization view (and stating that the two are the same problem is a plus signal).
- **SVD connection**: $X = U\Sigma V^T$ (centered X); principal directions = right singular vectors, component variances = $\sigma_i^2/(n-1)$; why SVD is the numerically preferred computation (never form $X^TX$; condition-number squaring); scores = $U\Sigma$.
- Practicalities: center always, scale when features have incommensurate units (correlation vs covariance PCA); explained-variance ratio and choosing #components; PCA as decorrelation; whitening; inability to capture nonlinear structure (kernel PCA one-paragraph pointer); PCA before k-means/kNN as a standard pipeline; PCA is unsupervised — top variance directions need not be predictive (a favorite trap probe).
- Interpretation caveats: loadings sign-indeterminacy; components as linear combos, not features.

### 3.5 t-SNE / UMAP caveats
Must-know (~2–3 pp, caveat-first framing):
- t-SNE mechanics at a sketch level: pairwise similarity matching between high-d Gaussians and low-d Student-t, KL objective, perplexity as an effective-neighborhood-size dial; why the heavy-tailed t distribution (crowding problem).
- **The caveat list interviewers probe**: cluster SIZES and inter-cluster DISTANCES in a t-SNE plot are not meaningful; global structure not preserved; results vary with perplexity/seed; t-SNE is for visualization, not a general-purpose feature transform (no natural out-of-sample mapping); can hallucinate clusters from noise.
- UMAP: faster, somewhat better global structure, has transform() for new points; still not distance-faithful. When to use PCA vs t-SNE/UMAP (linear + fast + invertible vs visualization).

### Whiteboard derivations
1. Centroid = mean minimizes SSE; Lloyd's convergence argument.
2. E and M steps of GMM in full (the chapter's centerpiece derivation).
3. PCA via constrained variance maximization → eigenproblem.
4. PCA–SVD equivalence.

### Common interview questions
- **'Derive the EM updates for a GMM.'** — Responsibilities by Bayes' rule; weighted-MLE M-step; then explain the ELBO argument for why likelihood monotonically increases.
- **'Why does k-means converge? To what?'** — Both steps decrease a bounded objective over finitely many partitions → converges; only to a local optimum; k-means++/restarts as mitigation.
- **'When does k-means fail? What do you use instead?'** — Enumerate the shape/density/outlier failures and map each to the fix (GMM, DBSCAN, spectral, k-medoids).
- **'What's the relationship between PCA and SVD?'** — Eigendecomposition of covariance vs SVD of centered data matrix; numerical-stability rationale.
- **'Your colleague reads distances off a t-SNE plot to argue two user segments are similar. Critique.'** — Inter-cluster distances/sizes are artifacts; check with metrics in the original/PCA space, vary perplexity/seed.
- **'Reduce 10K-dim sparse features for a downstream classifier — PCA, or something else?'** — TruncatedSVD (no centering, preserves sparsity), random projections, or just let a GBM/linear model with regularization handle it; discuss supervised alternatives.
- **'How is k-means a special case of GMM?'** — Isotropic equal covariances, $\sigma \to 0$ makes responsibilities one-hot → E-step = nearest centroid, M-step = mean.

### Red-flag wrong answers
- 'k-means converges to the global optimum.'
- Not centering before PCA, or computing PCA by explicitly inverting/eigendecomposing $X^TX$ on ill-conditioned data without noting SVD.
- 'PCA finds the most predictive features' (unsupervised ≠ predictive; also components ≠ features).
- EM presented as 'just alternate two steps' with no idea why likelihood increases.
- Reading t-SNE cluster sizes/distances literally.
- 'DBSCAN needs the number of clusters.'
- Choosing k purely by 'the elbow' with no acknowledgment of its subjectivity.

EXISTING COVERAGE: No DL-volume coverage. The closest touchpoints are embedding/retrieval material in sections/19_decision_frameworks.tex (ANN algorithms HNSW/IVF-PQ, lines ~330–345) — the new volume's kNN/ANN pointers (Ch. 1/8) and any 'clustering embeddings' aside should reference that table rather than redescribe ANN indexes. EM's ELBO framing can note a forward-reference to the DL volume's VAE/variational content if present there.

# Ch. 4 — Probabilistic Foundations: MLE/MAP/Bayes, Bias-Variance, Calibration, Uncertainty
**Target page weight: 10–12% of the volume (~18–22 pp).** This chapter supplies the vocabulary every other chapter leans on; keep it derivation-dense and cross-reference the DL volume aggressively for the parts it already owns (bias-variance identity, calibration mechanics).

### 4.1 MLE, MAP, full Bayes
Must-know:
- MLE: definition, log-likelihood, derive MLE for Bernoulli (count ratio) and Gaussian (sample mean; the biased $1/n$ variance estimator and why — a classic probe); properties at name-drop depth (consistency, asymptotic normality, invariance).
- MLE ↔ loss functions: **the unification table** — Gaussian noise → MSE, Laplace noise → MAE, Bernoulli → cross-entropy, Poisson → Poisson deviance; 'minimizing cross-entropy = MLE = minimizing KL to the empirical distribution' should be one derived line each.
- MAP: posterior ∝ likelihood × prior; **MAP = regularized MLE** — derive Gaussian prior → L2 and Laplace prior → L1 explicitly (this closes the loop with Ch. 1.2); prior strength ↔ regularization strength ↔ effective pseudo-counts (Beta-Bernoulli worked example: Laplace smoothing = Beta(2,2)-ish prior).
- Full Bayes: posterior predictive vs point estimates; conjugacy (Beta-Bernoulli, Gaussian-Gaussian at worked-example depth); when the distinction matters in interviews (small-data regimes, uncertainty-aware decisions, Thompson sampling — forward-ref Ch. 6); honest note that full Bayes is rarely the production path but the reasoning shows up in bandits and A/B analysis.

### 4.2 Bias-variance decomposition — derive it
Must-know:
- **The full derivation for squared loss**: add-and-subtract $\bar f = E[\hat f]$, expand, cross-terms vanish by independence of noise and estimator → $E[(y-\hat f)^2] = \text{Bias}^2 + \text{Var} + \sigma^2$, with every expectation's randomness source (training-set draw vs noise) made explicit — interviewers specifically check whether the candidate knows what the expectation is over.
- Classical knob table: which knobs move which term (k in kNN, depth in trees, λ in ridge, B and mtry in RF, rounds and lr in GBM) — tying every earlier chapter's hyperparameters to the decomposition is the value-add of this volume.
- Diagnosis: learning curves, train-vs-validation gap; what to do for high bias vs high variance.
- One paragraph: decomposition doesn't extend cleanly to 0-1 loss; and a two-line pointer to double descent / modern regime — **do not re-cover it** (owned by the DL volume).

### 4.3 Calibration and probabilistic evaluation
Keep to ~3 pp and cross-reference the DL volume for mechanics it already owns (ECE, reliability diagrams, temperature/Platt/isotonic). The classical value-add here:
- Which classical models are naturally calibrated-ish (logistic regression, by construction of the loss) vs systematically miscalibrated (naive Bayes overconfident due to independence violations; SVM has no probabilities; boosted trees push scores toward extremes; RF vote fractions compressed away from 0/1) — this model-by-model map is the classical interview question.
- Proper scoring rules: log loss and Brier as strictly proper; why optimizing a proper scoring rule incentivizes calibrated probabilities; accuracy/AUC are not proper.
- Platt scaling historically = fitting a sigmoid to SVM scores; isotonic needs more data.

### 4.4 Uncertainty
Must-know (~3 pp):
- **Aleatoric vs epistemic**, with a concrete example each and 'which shrinks with more data' (epistemic).
- Classical uncertainty tools: bootstrap CIs for any statistic (mechanics + when it fails — tiny n, heavy dependence), prediction intervals vs confidence intervals (the distinction is a frequent probe), quantile regression (pinball loss — derive why it yields quantiles), and **conformal prediction at 2026-interview depth**: split-conformal recipe, finite-sample marginal coverage guarantee, distribution-free; its rise makes it a differentiating name-drop that increasingly gets follow-up questions.
- Bayesian uncertainty pointer (posterior predictive) and ensemble disagreement as an epistemic proxy.

### Whiteboard derivations
1. MLE for Gaussian mean/variance; show the variance bias.
2. MAP with Gaussian prior → ridge (full derivation).
3. Bias-variance decomposition, all cross-terms shown vanishing.
4. Cross-entropy = MLE for Bernoulli; MSE = MLE under Gaussian noise.
5. Pinball loss minimizer = the τ-quantile.

### Common interview questions
- **'Derive the bias-variance decomposition.'** — The add-subtract expansion; strong answers name what each expectation averages over and give the knob table; cross-ref the DL volume for the deep-learning caveat.
- **'Show that ridge regression is MAP estimation.'** — Gaussian prior on weights, Gaussian likelihood, take negative log posterior → squared loss + L2; λ = σ²/τ².
- **'Why is the MLE of Gaussian variance biased? Does it matter?'** — Uses estimated mean, loses one df; matters little at large n; Bessel correction.
- **'MLE vs MAP vs fully Bayesian — when would you actually use each?'** — Data-size regimes, prior availability, decision-theoretic needs (Thompson sampling needs posteriors), compute.
- **'Your fraud model's scores are used to auto-decline transactions at a 0.9 threshold. What must be true of the scores?'** — Calibration, not just AUC; which models need post-hoc calibration; monitor calibration under drift.
- **'Confidence interval vs prediction interval?'** — CI: uncertainty of a parameter/mean; PI: uncertainty of a new observation (includes σ²); PI is wider; regression worked example.
- **'How would you give coverage-guaranteed prediction sets without distributional assumptions?'** — Split conformal: calibration-set residual quantile; marginal coverage; exchangeability caveat (breaks under drift/time series).

### Red-flag wrong answers
- Deriving bias-variance but unable to say what the expectation is taken over.
- 'MAP is Bayesian inference' (it's a point estimate; discards posterior uncertainty).
- 'Regularization is just a trick' with no prior interpretation available when probed.
- Conflating aleatoric and epistemic, or claiming more data fixes aleatoric noise.
- 'AUC 0.95 means the probabilities are trustworthy.'
- Prediction interval and confidence interval used interchangeably.
- Claiming the bias-variance trade-off means 'you can never reduce both' (better data/features/ensembling can).

EXISTING COVERAGE: Significant overlap to manage. sections/02_learning_theory.tex owns the bias-variance interview question (lines ~42–83): the identity, bias/variance definitions, the DL caveat (double descent, implicit regularization), red flags, and the diagnosis follow-up — the new volume should present the FULL derivation (which 02 states but does not derive step-by-step) plus the classical knob table, and explicitly cross-reference 02 for 'does it apply to deep learning'. sections/18_evaluation_metrics.tex owns calibration mechanics (lines ~121–145: ECE formula, reliability diagrams, Brier, temperature/Platt/isotonic, Guo et al.) and has a full miscalibration interview question (lines ~639–663) — the new volume should NOT restate ECE/fixes; its value-add is the per-classical-model calibration map and proper scoring rules. Bootstrap appears in 18 only as a significance-testing tool (lines ~480–502); uncertainty quantification (aleatoric/epistemic, conformal, quantile regression derivation) is net-new. MLE/MAP/Bayes is entirely net-new.

# Ch. 5 — The Applied Craft: Features, Leakage, Imbalance, CV, Interpretability
**Target page weight: 15–18% of the volume (~27–32 pp).** This is the chapter applied-company loops (Meta/Google/Stripe/Airbnb-style 'ML practical' rounds) weight most heavily after trees; it's also where personal notes are typically weakest (see diff checklist).

### 5.1 Feature engineering
Must-know:
- **Categorical encodings, with failure modes**: one-hot (cardinality explosion, collinearity/dummy trap for linear models), ordinal/label (fake order — fine for trees, wrong for linear), **target encoding and its leakage danger** (must be computed out-of-fold or with smoothing/noise; connect to CatBoost's ordered statistics, Ch. 2), frequency/count encoding, hashing trick (collisions as controlled noise; the ads-scale workhorse), learned embeddings as the NN-side pointer.
- Numerical transforms: log/Box-Cox for skew, binning (when it helps linear models, why it's mostly unnecessary for trees), scaling (standardize vs min-max vs robust; WHO needs it — distance/gradient/regularized methods — and who doesn't — trees), interactions and why linear models need them explicitly while trees/GBMs find (some) automatically.
- Datetime/cyclical features (sin/cos), aggregation features (user-level rollups and their leakage windows — forward-ref Ch. 7).
- **Missing data**: MCAR/MAR/MNAR taxonomy with one example each; deletion vs mean/median vs model-based (kNN/iterative) imputation; missingness-indicator features; why imputing before the split leaks; native NaN handling in XGBoost/LightGBM; the honest note that missingness is often informative (MNAR) and an indicator + native handling frequently beats clever imputation.

### 5.2 Leakage taxonomy (deserves its own numbered section — highest-value interview material)
Must-know, as a named taxonomy:
1. **Target leakage**: feature contains post-outcome information (e.g., 'number of late-payment calls' predicting default).
2. **Train-test contamination**: preprocessing fit on all data (scaler/imputer/encoder/feature-selection before split) — the canonical 'spot the bug' question.
3. **Temporal leakage**: random CV splits on time-ordered data; features computed over windows extending past the prediction time (cross-ref Ch. 7).
4. **Group leakage**: same user/patient/session in train and test (GroupKFold as fix; the Kaggle patient-ID war stories).
5. **Duplicate/near-duplicate leakage**.
6. **Label-derived features / proxy targets**.
Plus: the detection playbook — 'too good to be true' AUC, single dominant feature importance, performance collapse in production, feature-vs-target timestamp audit; and the fix: pipelines (fit-transform inside CV folds only).

### 5.3 Imbalanced learning
Must-know:
- Metrics first (cross-ref DL vol. ch. 18 for PR-AUC vs ROC-AUC — do not restate): the decision-order principle — fix the metric and threshold before touching the data.
- Threshold moving as the cheapest correct tool; cost-sensitive learning (class weights = loss reweighting; scale_pos_weight in XGBoost).
- Resampling: random over/undersampling; **SMOTE with its honest caveats** (synthetic points in feature space can be nonsense with categorical features/high-d; modern practice on tabular is usually class weights + threshold tuning rather than SMOTE — being 2026-honest here is a differentiator); undersampling + ensembling (EasyEnsemble-style).
- **Calibration consequence of resampling**: training on rebalanced data distorts predicted probabilities; the prior-correction formula or recalibration as the fix — frequently probed, rarely in personal notes.
- Extreme imbalance / anomaly-detection framing boundary (when 1e-5 prevalence stops being 'classification'); one-class methods and isolation forest at name-drop depth.

### 5.4 Model selection and CV — and its failure modes
Must-know:
- k-fold mechanics, stratification (when: classification, small data), leave-one-out (high variance of the estimate, cheap for linear via hat matrix), repeated CV.
- **The failure-mode catalog (the interview meat)**: (a) selection on the same folds used for reporting → optimistic bias → **nested CV** (outer for estimate, inner for tuning) with an honest note on when the extra cost is warranted (small data, high-stakes claims); (b) temporal data → expanding/rolling-origin splits, never shuffled k-fold (cross-ref Ch. 7); (c) grouped data → GroupKFold; (d) preprocessing outside folds (leakage §5.2); (e) many-hyperparameter search → the winner's-curse/overfitting-the-validation-set effect, 1-SE rule as a mitigation; (f) distribution shift between CV data and deployment (CV estimates in-distribution error only).
- Hyperparameter search: grid vs random (why random wins in high-d — Bergstra-Bengio argument), Bayesian/successive-halving at name-drop depth; early stopping's interaction with CV (which fold provides the stopping set).
- Final-model protocol: refit on all data with chosen hyperparameters; report the CV estimate, not the refit training score.

### 5.5 Interpretability: SHAP, permutation importance, and their traps
Must-know:
- The importance-method zoo: impurity-based (biased to high-cardinality/continuous; computed on train → reflects overfitting), permutation importance (model-agnostic, on held-out data; **trap: correlated features** — permuting one creates impossible feature combinations and splits credit unpredictably), drop-column retraining (expensive gold standard).
- **SHAP**: Shapley axioms at concept depth (efficiency/additivity — attributions sum to prediction minus baseline), TreeSHAP as the polynomial-time tree specialization, global = aggregated local; summary/dependence plots.
- **SHAP traps (the differentiating content)**: correlated features spread credit; interventional vs observational (conditional) value functions give different answers off-manifold; SHAP explains the MODEL, not the world — **no causal reading** ('high SHAP for feature X' ≠ 'changing X changes outcome'; cross-ref Ch. 6); background-dataset choice changes attributions; explanations can be unstable across retrains.
- Partial dependence plots and their independence assumption; ICE plots; monotonic constraints in XGBoost/LightGBM as 'interpretability by construction'; when regulation demands inherently interpretable models (scorecards, GAMs/EBM name-drop) vs post-hoc explanation.

### Common interview questions
- **'Our churn model has AUC 0.99 offline and 0.62 in production. Debug it.'** — Leakage taxonomy walk: timestamp audit of features, single-feature dominance, group contamination, temporal split check; strong answers ask 'when is each feature knowable?' first.
- **'How do you encode a 50K-cardinality categorical (e.g., merchant ID) for a GBM? For logistic regression?'** — Target encoding out-of-fold / CatBoost-native / hashing / embeddings; for LR: hashing + regularization; discuss leakage risk of target encoding explicitly.
- **'Fraud is 0.1% of transactions. Walk me through building the classifier.'** — Metric choice (PR-AUC, recall at fixed FPR — cross-ref DL vol.), class weights before resampling, threshold from cost matrix, calibration after any rebalancing, temporal validation.
- **'What is nested CV and when do you actually need it?'** — Inner loop tunes, outer loop estimates; needed when reporting an unbiased estimate while tuning on small data; overkill for large data with a clean held-out test set.
- **'Two features are highly correlated. What happens to permutation importance and SHAP?'** — Credit splitting/dilution in both; permutation additionally evaluates off-manifold points; fixes: group-permutation, drop-column, cluster-then-attribute.
- **'Why did SMOTE make your production fraud model worse?'** — Synthetic minority points off-manifold, distorted calibration, boundary noise amplification; class weights + threshold would have sufficed.
- **'PM asks: SHAP says discount_rate drives conversion — should we raise discounts?'** — Explanation ≠ causation; confounding; propose an experiment (bridge to Ch. 6).

### Red-flag wrong answers
- Scaling/imputing/encoding on the full dataset before splitting, presented without hesitation.
- 'Use SMOTE' as the reflexive first answer to imbalance (before metrics/weights/threshold).
- Not knowing target encoding leaks without out-of-fold computation.
- 'Accuracy dropped so retrain' with no leakage/drift differential diagnosis.
- Treating impurity importance or SHAP as causal effects.
- Random k-fold on time series (also covered in Ch. 7 red flags).
- 'Nested CV' recited but unable to say which loop does what, or why plain CV is biased after tuning.
- Claiming trees need one-hot for all categoricals, or that missing values must always be imputed.

EXISTING COVERAGE: sections/18_evaluation_metrics.tex owns imbalance METRICS (accuracy warning box line ~88, ROC-vs-PR-AUC table + keyinsight lines ~91–119) — the new volume covers imbalance TREATMENT (weights, resampling, thresholds, calibration-after-resampling) and cross-references 18 for metric choice. 17_production_systems.tex covers training-serving skew (lines ~217+) and drift/PSI (lines ~282–321), which border the leakage/debugging material — the new volume's leakage taxonomy is about train-time contamination and should point to 17 for serving-time skew and drift monitoring. Feature stores/batch-vs-streaming features are owned by 17 (§ lines ~179–216); interpretability (SHAP/permutation), CV/nested-CV, encodings, missing data, and the leakage taxonomy have no DL-volume coverage — net-new.

# Ch. 6 — Experimentation and Causal Inference: A/B Beyond Basics, Uplift, Bandits, Observational Methods
**Target page weight: 10–12% of the volume (~18–22 pp).** Platform companies (Meta, Airbnb, Uber, DoorDash, Netflix) test this hard in applied-scientist and MLE loops; frontier labs test it lightly. The DL volume already owns A/B basics and significance testing — this chapter must start where those end.

### 6.1 A/B beyond the basics
Must-know (assume the reader has DL vol. ch. 17/18 basics: sample-size formula, power, peeking, duration, multiple comparisons):
- **Variance reduction**: CUPED derived properly — regression-adjust the metric with its pre-experiment value, $Y' = Y - \theta(X - \bar X)$, variance shrinks by $1-\rho^2$; when it fails (new users, no pre-period); stratification/post-stratification as the simpler cousin. (18 name-drops CUPED twice; this volume owns the derivation.)
- **Interference / SUTVA violations**: network effects (social products), marketplace cannibalization (two-sided platforms); designs that address them — cluster randomization (and its variance cost), switchback (own the details 17's table only sketches: randomization unit = region×time-bucket, carryover bias, burn-in periods), budget-split for ads.
- Sequential testing done right: group-sequential/alpha-spending vs always-valid (mSPRT) at concept depth — 17's warning box says 'don't peek'; this volume explains what to use when you must monitor.
- Ratio metrics and the delta method (per-user vs per-event randomization unit mismatch — CTR = clicks/impressions randomized by user); heterogeneous treatment effects and why subgroup fishing needs pre-registration or HTE methods (leads into uplift).
- Practical pathologies: sample-ratio mismatch (SRM) as the #1 trust check, novelty/primacy effects, Twyman's law.

### 6.2 Uplift modeling (CATE)
Must-know:
- The four quadrants (persuadables/sure things/lost causes/sleeping dogs); why churn-risk targeting ≠ uplift targeting (you may target sure-things or trigger sleeping dogs) — the canonical motivating story.
- Meta-learners: S-learner (treatment as a feature; regularization bias toward zero effect), T-learner (two models; differencing amplifies noise), X-learner (small treatment groups), and doubly-robust/DR-learner + causal forest at name-drop depth.
- Evaluation without ground-truth effects: uplift/Qini curves, why standard CV on outcomes doesn't validate uplift; requires randomized data (or strong assumptions).

### 6.3 Bandits
Must-know:
- Explore-exploit; regret as the objective; ε-greedy, UCB (optimism + the confidence-bound intuition; UCB1 formula), **Thompson sampling** (posterior sampling; Beta-Bernoulli worked example — connects to Ch. 4 conjugacy); when bandits beat A/B (many arms, perishable content, cheap reward signal) and when they don't (need unbiased effect estimates, guardrails, delayed rewards).
- Contextual bandits at applied depth: LinUCB sketch; off-policy evaluation via inverse propensity scoring — one derived paragraph, since IPS is name-dropped in the DL volume's offline-online-gap discussion and interviewers follow up; the recommendation/exploration tie-in.
- Practical issues: delayed feedback, non-stationarity, batch updates.

### 6.4 Observational causal inference at interview depth
Must-know (each method: assumptions, estimator sketch, one failure mode — NOT textbook depth):
- Framework: potential outcomes, ATE/ATT/CATE, the fundamental problem; confounding; **the assumptions that do all the work** — ignorability/unconfoundedness, positivity/overlap, SUTVA; 'no unmeasured confounding is untestable' as the honest headline.
- **Propensity scores**: definition, IPW estimator and its variance blow-up near 0/1 propensities (trimming/stabilized weights), matching, checking overlap and balance (SMD), doubly-robust (AIPW) at concept depth — 'consistent if either model is right'.
- **Difference-in-differences**: parallel-trends assumption (and pre-trend checks), the 2×2 estimator, one failure mode (differential shocks/composition change); event-study framing name-drop.
- **Instrumental variables**: relevance + exclusion restriction, the Wald/2SLS estimator sketch, weak-instrument danger, LATE caveat ('effect for compliers only'); encouragement designs as the industry-realistic IV example.
- Regression discontinuity in one paragraph (sharp vs fuzzy, bandwidth sensitivity).
- A closing decision guide: 'experiment if you can; otherwise which quasi-experiment fits which data shape' + the ML tie-in (why naive counterfactual claims from predictive models fail; cross-ref Ch. 5.5 SHAP-isn't-causal).

### Common interview questions
- **'Your A/B test on a marketplace shows +2% GMV in treatment. Why might the true effect be smaller (or negative)?'** — Cannibalization/interference: treatment steals from control through shared supply; switchback or cluster randomization; also novelty effects, SRM check first.
- **'Derive how CUPED reduces variance. When does it fail?'** — Regression adjustment with pre-period covariate; optimal θ = cov/var; variance × (1−ρ²); fails for new users/weak pre-period correlation.
- **'We can't randomize prices. How do you estimate price elasticity from logs?'** — Confounding story (prices set in response to demand); candidate proposes IV (cost shocks), DiD (staggered rollouts), or RD (pricing-rule thresholds), states assumptions and their fragility; best answers still push for a small experiment.
- **'Why not target retention offers at users with highest churn probability?'** — Uplift vs risk; persuadables vs sure things/sleeping dogs; T/X-learner + Qini evaluation on a randomized holdout.
- **'Thompson sampling vs UCB vs ε-greedy — mechanics and when each?'** — Posterior sampling vs optimism vs naive; TS's empirical strength and easy batch/delay handling; ε-greedy's constant regret.
- **'When would you choose a bandit over an A/B test?'** — Regret minimization vs inference; many arms/short-lived content favor bandits; need for precise, guardrailed effect estimates favors A/B (17's table gives the one-liner; this volume gives the reasoning).
- **'What assumption does DiD need, and how do you probe it?'** — Parallel trends; pre-period placebo/event-study plot; discuss a concrete violation.

### Red-flag wrong answers
- 'Add the confounder as a regression control' as a universal fix, with no mention of unmeasured confounding or overlap.
- Believing propensity scores fix unobserved confounding.
- Running a user-randomized test on a two-sided marketplace with no mention of interference.
- 'Bandits are always better than A/B because they waste less traffic' (loses clean inference/guardrails).
- Uplift model evaluated by outcome AUC.
- Unable to name the IV exclusion restriction, or presenting IV without any caveat.
- Peeking/optional stopping treated as fine 'because we use p<0.05' (cross-ref 17's warning; the red flag is not knowing sequential corrections exist).
- Reading a causal conclusion off SHAP/feature importance (shared with Ch. 5).

EXISTING COVERAGE: Substantial adjacent coverage to build on, not repeat. sections/17_production_systems.tex §A/B Testing (lines ~238–280) owns: basics, the 16p(1-p)/δ² rule of thumb, duration/power one-liners, the peeking warning box (always-valid p-values name-dropped), and the alternatives TABLE (bandit/interleaving/switchback with one-line when/limitation) plus a paragraph each on interleaving and bandits (Thompson/UCB name-dropped). sections/18_evaluation_metrics.tex owns significance testing (bootstrap, paired tests, McNemar), multiple comparisons (Bonferroni/BH), power analysis + the worked sample-size question (lines ~691–715, which name-drops CUPED and sequential testing), and the offline-online gap question (lines ~743–767, which name-drops inverse propensity scoring). The new volume must start PAST these: derive CUPED, explain sequential methods rather than name them, give switchback/cluster designs real treatment, and own uplift, bandit algorithms in detail, and all observational causal inference — none of which exist in the DL volume.

# Ch. 7 — Time Series Essentials: Classical, Gradient-Boosted, and When Deep Learning
**Target page weight: 6–8% of the volume (~11–15 pp).** A minority topic in most loops but a mainstay at fintech, forecasting-heavy platforms (demand/supply, capacity), and marketplaces; also the richest source of validation-leakage traps, which every company asks about.

### 7.1 Foundations
Must-know:
- Components: trend/seasonality/cycle/noise; additive vs multiplicative decomposition (STL name-drop); autocorrelation (ACF/PACF reading at 'identify AR vs MA signatures' depth).
- **Stationarity**: definition (weak stationarity: constant mean/variance/autocovariance), why models assume it, tests at name-drop depth (ADF), transformations to achieve it (differencing, log, detrending); spurious regression between trending series as the classic trap.
- Baselines that must be beaten and reported: naive (last value), seasonal naive, drift — interviewers reward candidates who insist on these before any model.
- Forecast evaluation: MAE/RMSE/MAPE-and-its-zeros (cross-ref DL vol. ch. 18's MAPE warning), sMAPE, **MASE** (scaled by naive baseline — the interview-differentiating metric), pinball loss for quantile forecasts (cross-ref Ch. 4.4); forecast horizon and error growth.

### 7.2 Classical models
Must-know (working depth, not textbook depth):
- Exponential smoothing family: SES → Holt (trend) → Holt-Winters (seasonality); ETS as the honest strong baseline that wins many real forecasting jobs.
- ARIMA: AR/I/MA components and what each models; order selection via ACF/PACF or auto-ARIMA/AIC; SARIMA for seasonality; one worked intuition ('AR(1) = mean-reverting with memory φ').
- One paragraph each: exogenous regressors (ARIMAX/dynamic regression), Prophet-style decomposable models (and the honest note that they're a convenience, frequently beaten by ETS/GBM).

### 7.3 ML on time series (the GBM recipe)
Must-know — this subsection is the applied-interview core:
- **Feature recipe**: lags, rolling-window statistics (with windows ending BEFORE the prediction origin), calendar/cyclical features, holiday/event flags, target transforms; the global-model pattern (one GBM across thousands of related series with series-ID/hierarchy features — the M5 lesson).
- **Leakage discipline**: every feature answerable to 'was this knowable at prediction time?'; centered rolling windows, future-fit scalers, and shuffled CV as the standard bugs (cross-ref Ch. 5.2 taxonomy, temporal entry).
- **Validation**: rolling-origin/expanding-window backtesting (diagram required), gap/embargo between train and test when features have lookahead windows, why random k-fold is invalid; multi-horizon strategy: recursive vs direct forecasting trade-offs (error accumulation vs per-horizon models).
- **The trees-can't-extrapolate problem**: GBMs cap at training-range values → detrend/difference the target, predict deltas, or hybrid (linear trend + GBM residuals) — a favorite probe tying back to Ch. 2.
- Probabilistic forecasts via quantile-loss GBMs.

### 7.4 When deep learning
Keep to 1–2 pp, judgment-focused, cross-referencing the DL volume for architecture detail:
- DL earns its complexity when: very many related series (global models with shared learning), rich exogenous/multimodal inputs, long-range or complex seasonal interactions, need for learned embeddings of categorical hierarchy (DeepAR/N-BEATS/TFT and 2025-era foundation models like TimesFM/Chronos at name-drop depth).
- Classical/GBM wins when: few series, short history, strong simple seasonality, interpretability/ops constraints; the M-competition headline that simple methods and GBMs beat sophisticated DL far more often than papers imply.
- A short decision table (series count × history length × exogenous richness → ETS/ARIMA vs GBM vs DL).

### Common interview questions
- **'Forecast daily demand for 5K SKUs. Walk me through your approach.'** — Baselines first, global GBM with lag/rolling/calendar features, rolling-origin backtest with MASE, hierarchy reconciliation name-drop, cold-start SKUs; escalate to DL only with cause.
- **'Why can't you use standard k-fold CV on time series, exactly?'** — Temporal leakage: future information in training folds, autocorrelation makes 'independent folds' false; rolling-origin + embargo; quantify the optimism with the AUC-drop story.
- **'Your GBM forecast flatlines at the recent maximum while demand is growing. Why?'** — Piecewise-constant trees can't extrapolate; predict differenced target or hybrid trend model.
- **'What is stationarity and why do ARIMA-type models want it?'** — Stable moments make estimated autocorrelations meaningful; differencing; spurious-regression example.
- **'Recursive vs direct multi-step forecasting?'** — One model fed its own predictions (error compounding, consistent dynamics) vs one model per horizon (no compounding, more models, jagged horizons); ensembles of both.
- **'How do you build a forecast with uncertainty bands the business can trust?'** — Quantile GBM/pinball loss or conformal-on-residuals (cross-ref Ch. 4.4); backtest coverage empirically, not just nominally.

### Red-flag wrong answers
- Shuffled/random CV on temporal data (the single most common practical red flag in this area).
- Rolling features computed with centered or future-inclusive windows.
- Reaching for LSTM/transformer before naming a naive baseline ('DL-first' on 200 data points).
- MAPE on a series with zeros; comparing MAE across series of different scales without MASE.
- Not knowing trees can't extrapolate trend.
- Fitting ARIMA to an obviously non-stationary series and reporting in-sample fit as skill.
- Claiming Prophet 'handles everything automatically'.

EXISTING COVERAGE: Thin, deliberately: sections/19_decision_frameworks.tex's architecture table (line ~59) has one row — 'Time series: LightGBM + features; alternative Transformer/LSTM when long-range patterns' — which the new chapter expands into its §7.3/§7.4 with reasoning behind that row. sections/18_evaluation_metrics.tex owns the MAPE warning box (lines ~271–273, zeros + asymmetry) and general regression metrics — cross-reference for those, add MASE/pinball here. Temporal drift appears in 17_production_systems.tex's drift section (concept vs data drift, lines ~288–321) — reference for the monitoring side. All modeling content (stationarity, ETS/ARIMA, the GBM feature/backtest recipe, extrapolation, multi-horizon) is net-new.

# Ch. 8 — ML Coding Round Implementables: The From-Scratch Canon with Grading Rubrics
**Target page weight: 12–15% of the volume (~22–27 pp).** Coding rounds at both frontier labs and platform companies still draw from a small canonical set. Format per algorithm: (a) reference implementation in clean NumPy (~30–60 lines), (b) the design decisions an interviewer probes, (c) complexity analysis, (d) test cases/edge cases the candidate should volunteer, (e) a grading rubric (junior/senior/staff bars), (f) common bugs. General round advice up front: clarify the spec (fit/predict API? vectorized? edge cases?), write the API skeleton first, narrate complexity, test with a tiny hand-checkable example.

### 8.1 k-means (the #1 ask)
- Reference: init (random choice from data, then k-means++ as the upgrade), assignment via vectorized pairwise distances (`||a-b||² = ||a||² + ||b||² − 2ab` trick), update, convergence check (assignment stability or centroid shift < tol), empty-cluster handling (reseed from farthest point).
- Probes: why k-means++ (derive D² sampling), complexity O(nkd·iters), local optima/restarts, when it fails (cross-ref Ch. 3.1).
- Rubric — junior: correct loop, may be unvectorized, handles the happy path. Senior: vectorized distances, k-means++, empty-cluster and convergence handling, states complexity unprompted. Staff: discusses minibatch variant, numerical concerns of the expansion trick (negative values from cancellation → clip), ties failure modes to init and data shape, writes a test with a known 2-blob answer.
- Common bugs: forgetting to recompute after last update, argmin over the wrong axis, empty-cluster NaN mean, comparing floats for convergence with ==.

### 8.2 Logistic regression with SGD
- Reference: sigmoid with overflow guard (clip logits or use the `np.where` stable form / logaddexp for the loss), BCE loss, gradient $X^T(p-y)/n$, minibatch SGD loop with shuffling each epoch, optional L2, bias via appended column or separate term.
- Probes: derive the gradient (cross-ref Ch. 1.3), why the stable sigmoid/log-sum-exp matters (log(0) at p=1), learning-rate behavior, full-batch vs minibatch, how you'd verify (loss monotone-ish decrease; gradient check by finite differences — the staff-level move).
- Rubric — junior: correct forward/backward, trains on toy data. Senior: numerical stability handled, shuffling, L2, vectorized, gradient derivation on request. Staff: finite-difference gradient check, discusses separation divergence without regularization, class-weight extension, convergence criteria.
- Common bugs: exp overflow, missing 1/n, sign errors, forgetting to shuffle, labels {−1,1} vs {0,1} mismatch.

### 8.3 Decision tree split finding (usually 'best split for one node' or a small full tree)
- Reference: Gini/entropy functions; for each feature, sort once and sweep thresholds updating class counts incrementally (the O(n log n + n·classes) scan vs the naive O(n²) re-partition — THE probe of this exercise); weighted-impurity gain; recursion with depth/min-samples stopping; majority-vote leaves.
- Probes: why midpoints of sorted uniques, why the incremental-count sweep, greedy vs optimal, how regression changes it (variance via running sums — derive the O(1) update from sum and sum-of-squares), categorical features.
- Rubric — junior: correct gain computation, naive threshold loop acceptable. Senior: sorted incremental sweep, correct handling of ties/constant features, clean recursion, complexity stated. Staff: connects to histogram binning in LightGBM (Ch. 2.4), discusses numerical care in variance-by-sums, handles min_impurity_decrease and pruning discussion.
- Common bugs: evaluating split at every value including duplicates, off-by-one in left/right counts, impurity of empty side, no stopping → infinite recursion on identical points.

### 8.4 One boosting round (gradient boosting on squared or log loss, stumps as weak learners)
- Reference: init F₀ (mean or log-odds prior), loop: compute pseudo-residuals (−gradient), fit a stump/shallow tree to residuals (reuses 8.3), optionally line-search or use the leaf-value formula, update F += η·h; staged predictions for plotting train/val loss.
- Probes: pseudo-residuals for log loss = y−p (derive), why shrinkage, early stopping, what changes for second-order (XGBoost leaf value g/h — cross-ref Ch. 2.4); 'implement the leaf-weight formula' as the senior extension.
- Rubric — junior: correct squared-loss residual loop with a provided tree learner. Senior: log-loss version with derived residuals, shrinkage + early stopping, monotone train-loss sanity test. Staff: second-order leaf weights, discusses why fitting h to residuals ≈ functional gradient step, subsample option, connects rounds/lr trade.
- Common bugs: fitting to y instead of residuals after round 1, missing learning rate (overfits instantly), log-loss version updating in probability space instead of logit space.

### 8.5 PCA via power iteration
- Reference: center (probe: why), power iteration on covariance (or on Xᵀ(Xv) to avoid forming the d×d matrix — the probe), normalize each step, convergence via |v·v_prev|→1, eigenvalue via Rayleigh quotient, deflation (subtract λvvᵀ or project out) for subsequent components; verify against np.linalg.svd.
- Probes: why power iteration converges to the top eigenvector (ratio (λ₂/λ₁)^t argument — sketchable), failure when λ₁≈λ₂, why not form XᵀX explicitly for wide data, orthogonality drift after many deflations.
- Rubric — junior: correct centered power iteration for PC1. Senior: matrix-free X-only products, deflation for k components, convergence criterion, SVD cross-check test. Staff: convergence-rate argument, degenerate-eigenvalue caveat, mentions randomized SVD as the production method, explained-variance reporting.
- Common bugs: forgetting to center (silently wrong PC1), not normalizing per iteration (overflow), deflating with unnormalized vectors, sign flipping treated as non-convergence.

### 8.6 kNN with kd-tree flavor
- Reference tier 1 (always): brute-force vectorized kNN — pairwise-distance trick, argpartition for top-k (probe: why argpartition O(n) beats full sort), majority vote / mean, standardization reminder.
- Reference tier 2 (the 'flavor'): kd-tree build (median split on cycling/widest-spread axis, O(n log n)) and nearest-neighbor search with the hypersphere-vs-splitting-plane pruning test (|query_axis − split| < current_best → must visit the far side); best-first k-NN via bounded max-heap.
- Probes: why kd-trees degrade in high dimensions (pruning stops firing; ~d>20 rule of thumb — ties to curse of dimensionality, Ch. 1.6), expected vs worst-case query complexity, what production uses instead (HNSW/IVF — cross-ref DL vol. ch. 19 ANN table).
- Rubric — junior: correct brute force with argpartition. Senior: kd-tree build + correct single-NN search with pruning explained. Staff: k-NN heap variant, high-d degradation analysis, articulates exact-vs-approximate trade and when brute force on GPU is actually the right answer.
- Common bugs: pruning test using squared vs non-squared distance inconsistently, forgetting the current node itself as a candidate, unbalanced builds from sorting instead of median-partition (np.partition), ties in voting.

### Also-plausible extras (one paragraph + skeleton each, no full treatment)
Linear regression via normal equations AND via GD (asked as a warm-up), naive Bayes text classifier with log-space + smoothing, k-fold CV harness from scratch (splitting without sklearn — tests indexing care), AUC from scratch (rank-based O(n log n) formulation — bridges to DL vol. ch. 18), and a train/test split with stratification.

### Chapter-wide grading rubric (what interviewers actually score)
1. **Correctness on edge cases** (empty cluster, single class, constant feature, k>n) — volunteered, not prompted.
2. **Vectorization and complexity fluency** — states big-O unprompted; no accidental O(n²) loops where a sort suffices.
3. **Numerical hygiene** — stable sigmoid/log-sum-exp, centering, tolerance-based convergence.
4. **API/test discipline** — fit/predict shape, a hand-checkable test written unasked.
5. **Connecting code to theory** — 'this residual is the negative gradient because…' is what separates staff from senior.

### Red-flag performances
- Reciting sklearn API instead of implementing; unable to write the impurity or sigmoid from memory.
- No convergence criterion (fixed 10 iterations, no rationale).
- exp overflow crash on the interviewer's first adversarial input.
- O(n² d) split finding with no awareness a sort-based sweep exists.
- Cannot connect the implementation to the math when asked ('why is this the gradient?').
- Claims the code works without running/testing anything on a small example.

EXISTING COVERAGE: No DL-volume coverage of from-scratch classical implementations (the DL volume has no coding-round chapter for classical algorithms). Touchpoints to cross-reference rather than duplicate: the ANN-index table in sections/19_decision_frameworks.tex (lines ~330–345) for the 'what production uses instead of kd-trees' probe, and sections/18_evaluation_metrics.tex for AUC's ranking interpretation (line ~109) behind the AUC-from-scratch extra. Entirely net-new otherwise.

# Diff Checklist — What Personal Conventional-ML Notes Typically Contain vs. Miss
Run this against the user's document on arrival. Structure: for each area, **[USUALLY PRESENT]** items to verify exist and are correct, and **[USUALLY MISSING]** items to check for and flag as gaps — the missing list is ordered by interview ROI within each area. Then a scoring pass and page-weight audit.

### A. Supervised core
[USUALLY PRESENT — verify correctness, don't re-add]: linear/logistic definitions, sigmoid, L1-vs-L2 one-liner ('L1 sparsity'), bias-variance one-liner, SVM 'maximum margin' phrase, kernel trick phrase, kNN description, overfitting/underfitting.
[USUALLY MISSING — flag if absent]:
- ☐ Normal-equations DERIVATION + complexity + why libraries use QR/SVD not inversion
- ☐ Soft-thresholding math behind L1 sparsity (not just the diamond picture)
- ☐ Logistic gradient derived from the likelihood; the (p−y) form; separation → divergence
- ☐ GLM/exponential-family unification (almost never in personal notes)
- ☐ SVM dual actually derived (KKT, why kernels enter); most notes stop at the margin picture
- ☐ Honest 'when is each model still the right choice in 2026' judgment notes
- ☐ Generative-vs-discriminative with the NB/LR sample-efficiency asymptotics
- ☐ Curse-of-dimensionality argument with actual reasoning (distance concentration), not just the phrase

### B. Trees/ensembles
[USUALLY PRESENT]: Gini/entropy formulas, 'RF = bagging + feature subsampling', 'boosting fits residuals', an XGBoost hyperparameter list.
[USUALLY MISSING]:
- ☐ Functional-gradient-descent framing (residual-fitting shown as the squared-loss special case; log-loss pseudo-residuals y−p)
- ☐ XGBoost second-order derivation: leaf weight −G/(H+λ) and the gain formula (the single highest-ROI gap in most notes)
- ☐ Correlated-average variance formula ρσ² + (1−ρ)σ²/B motivating mtry
- ☐ OOB and the 1−1/e derivation
- ☐ Leaf-wise vs level-wise growth; histogram split finding; GOSS/EFB
- ☐ CatBoost ordered target statistics ↔ leakage connection
- ☐ Why Gini beats misclassification error for splitting (concavity)
- ☐ Trees-can't-extrapolate and its time-series consequence

### C. Unsupervised
[USUALLY PRESENT]: k-means algorithm steps, elbow method, PCA 'reduces dimensionality', maybe a DBSCAN mention.
[USUALLY MISSING]:
- ☐ Full GMM E/M-step derivation + why EM increases likelihood (ELBO) — the most commonly absent derivation in personal notes
- ☐ k-means++ mechanics and the k-means↔GMM limit connection
- ☐ k-means failure-mode enumeration mapped to alternatives
- ☐ PCA derived (variance-max → eigenproblem) and the SVD connection with the numerical rationale
- ☐ 'PCA components ≠ predictive features' trap
- ☐ t-SNE/UMAP caveat list (distances/sizes meaningless, perplexity sensitivity, no out-of-sample for t-SNE)

### D. Probabilistic foundations
[USUALLY PRESENT]: MLE definition, Bayes' rule, bias-variance formula stated.
[USUALLY MISSING]:
- ☐ Bias-variance actually DERIVED with explicit expectation sources
- ☐ MAP→regularization derivations (Gaussian→L2, Laplace→L1)
- ☐ Loss-function↔noise-model table (MSE=Gaussian MLE etc.)
- ☐ Per-model calibration map (which classical models are/aren't calibrated and why)
- ☐ Proper scoring rules
- ☐ Aleatoric/epistemic distinction; prediction vs confidence intervals
- ☐ Conformal prediction (2026-relevant; almost never in older notes)

### E. Applied craft
[USUALLY PRESENT]: one-hot/label encoding, 'use cross-validation', SMOTE mention, 'SHAP for interpretability', scaling reminder.
[USUALLY MISSING]:
- ☐ A NAMED leakage taxonomy (target/contamination/temporal/group/duplicate) + detection playbook — the highest-ROI applied gap
- ☐ Target encoding's out-of-fold requirement
- ☐ Missingness taxonomy (MCAR/MAR/MNAR) and indicator-features guidance
- ☐ Imbalance decision ORDER (metric → weights/threshold → maybe resample) and the calibration-distortion of resampling; honest SMOTE skepticism
- ☐ Nested CV: what it fixes, which loop does what, when it's overkill
- ☐ CV failure modes: winner's curse over big searches, group leakage, temporal splits
- ☐ Permutation-importance and SHAP traps (correlated features, off-manifold, not causal)
- ☐ Grid-vs-random-search rationale

### F. Experimentation/causal
[USUALLY PRESENT]: A/B definition, p-value, maybe a bandit sentence. (Many personal ML notes have essentially nothing here — treat an empty section as the expected finding.)
[USUALLY MISSING]:
- ☐ CUPED derivation; SRM check; ratio-metric/delta-method issue
- ☐ Interference/marketplace designs (cluster, switchback) in more than one line
- ☐ Sequential testing methods (not just 'don't peek')
- ☐ Uplift: four quadrants, S/T/X-learners, Qini evaluation
- ☐ Thompson sampling with the Beta-Bernoulli mechanics; off-policy eval/IPS
- ☐ Potential outcomes + the assumption trio (ignorability, positivity, SUTVA)
- ☐ Propensity/IPW with the variance-blow-up caveat; DiD parallel trends; IV exclusion + LATE caveat

### G. Time series
[USUALLY PRESENT]: ARIMA acronym expansion, 'stationarity', maybe train/test split note.
[USUALLY MISSING]:
- ☐ Rolling-origin backtesting with embargo (and the explicit random-k-fold prohibition)
- ☐ The GBM lag/rolling/calendar feature recipe and global-model pattern
- ☐ Trees-can't-extrapolate fix (differencing/hybrid)
- ☐ MASE; recursive-vs-direct multi-step
- ☐ Naive/seasonal-naive baselines as mandatory comparators
- ☐ Honest classical-vs-GBM-vs-DL decision guidance

### H. Coding implementables
[USUALLY PRESENT]: perhaps a k-means or logistic-regression snippet copied from a course.
[USUALLY MISSING]:
- ☐ The full 6-algorithm canon (k-means, LR+SGD, tree split, boosting round, PCA power iteration, kNN/kd-tree)
- ☐ Numerical-stability idioms (stable sigmoid, distance-trick clipping, centering)
- ☐ Sorted incremental split-sweep (vs naive O(n²))
- ☐ Edge-case inventory per algorithm (empty cluster, separation, constant feature)
- ☐ Self-testing habit: hand-checkable examples, finite-difference gradient check, SVD cross-check
- ☐ Complexity annotations on every implementation

### Cross-cutting audit passes (run after the per-area diff)
1. **Derivation depth**: for each of the 12 whiteboard derivations named in Chs. 1–4 (normal equations, logistic gradient, L1 soft-thresholding, SVM dual, correlated-variance, 1/e, functional GD, XGBoost leaf/gain, EM, PCA eigen + SVD, bias-variance, MAP→reg), classify the user's coverage as absent / stated / derived. Anything below 'derived' on: bias-variance, EM, XGBoost leaf value, logistic gradient — flag as priority-1.
2. **Judgment content**: does each model section answer 'when would you still use this in 2026, and what replaced it'? Notes that are all mechanics and no judgment fail breadth rounds.
3. **Red-flag inoculation**: check whether the notes explicitly warn against the ~40 red-flag answers listed across Chs. 1–8; personal notes almost never encode negative knowledge.
4. **Interview-question coverage**: map the ~50 sketched questions across chapters against the notes; expect <40% hit rate in typical personal notes, concentrated in Chs. 1–3.
5. **Page-weight audit** against this spec's targets: Ch1 14–18%, Ch2 18–22%, Ch3 12–15%, Ch4 10–12%, Ch5 15–18%, Ch6 10–12%, Ch7 6–8%, Ch8 12–15%. Typical personal notes are overweight Ch1/Ch3 theory and underweight Ch2 internals, Ch5, Ch6, and Ch8 — expect the diff to recommend reallocating toward trees-internals, applied craft, and coding.
6. **Duplication-with-DL-volume check**: flag any sections in the user's notes that restate what the DL volume already owns (bias-variance-in-DL discussion, ECE/calibration fixes, ROC-vs-PR mechanics, A/B basics/sample-size, GBM-vs-NN question, ANN indexes, drift/PSI) and convert them to cross-references per the existing_coverage notes on each chapter above.

EXISTING COVERAGE: This section is the diff harness itself; its duplication-check pass (item 6) encodes the four reviewed files' ownership: sections/02_learning_theory.tex (bias-variance identity + DL caveats), sections/18_evaluation_metrics.tex (classification/ranking/regression metrics, calibration mechanics, significance testing, power analysis, offline-online gap), sections/17_production_systems.tex (A/B basics + peeking + bandit/interleaving/switchback table, drift/PSI, training-serving skew, feature stores), sections/19_decision_frameworks.tex (GBM-vs-NN question, architecture tables, ANN index table, start-simple ladder).

