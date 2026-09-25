# Theory and methods

This document collects the mathematics behind every component of `critmc`:
what is computed, why each algorithm samples the right distribution, and how
the statistical analysis turns correlated Monte Carlo data into estimates with
honest error bars. References are listed at the end; equation numbers are
local to each section.

Units: $J = k_B = 1$ throughout, $\beta = 1/T$, $N$ is the number of sites,
$L$ the linear size, $d$ the dimension.

---

## 1. Models

All models live on periodic lattices described by a neighbour table (see
`critmc.lattice`). The bond list contains every nearest-neighbour pair once.

| model | variables | Hamiltonian |
|---|---|---|
| Ising | $s_i \in \{-1, +1\}$ | $H = -\sum_{\langle ij\rangle} s_i s_j - h\sum_i s_i$ |
| $q$-state Potts | $s_i \in \{0,\dots,q-1\}$ | $H = -\sum_{\langle ij\rangle}\delta_{s_i s_j}$ |
| O($n$) vector | $\mathbf S_i \in S^{n-1}$ | $H = -\sum_{\langle ij\rangle}\mathbf S_i\cdot\mathbf S_j$ |
| Edwards–Anderson | $s_i \in \{-1,+1\}$ | $H = -\sum_{\langle ij\rangle} J_{ij}s_i s_j$, $J_{ij}\sim\mathcal N(0,1)$ |

$n = 2$ is the XY model and $n = 3$ the classical Heisenberg model. The
2-state Potts model is the Ising model with $K_\text{Ising} = K_\text{Potts}/2$,
which the test suite uses as a cross-check.

**Order parameters.** Ising: $m = N^{-1}\sum_i s_i$. Potts: with site
densities $\rho_k = n_k/N$, the vector order parameter in the simplex
representation has
$$ |m|^2 = \frac{q\sum_k \rho_k^2 - 1}{q-1}, $$
which is 1 in a fully ordered state and 0 when all states are equally
populated. O($n$): $\mathbf m = N^{-1}\sum_i \mathbf S_i$.

**Response functions.** Per site,
$$ c = \frac{\beta^2}{N}\left(\langle E^2\rangle - \langle E\rangle^2\right),\qquad
\chi = \beta N\left(\langle m^2\rangle - \langle |m|\rangle^2\right),\qquad
\chi_0 = \beta N \langle m^2\rangle . $$
On a finite lattice $\langle m\rangle = 0$ by symmetry, so the connected
susceptibility uses $\langle|m|\rangle$; $\chi_0$ is the natural quantity at and
above $T_c$.

**Binder cumulant.**
$U_4 = 1 - \langle m^4\rangle / (3\langle m^2\rangle^2)$ tends to $2/3$ in an
ordered phase and to 0 in a disordered Ising phase. It is dimensionless, so at
$T_c$ it approaches a universal (boundary-condition and shape dependent) value
$U^*$; curves for different $L$ cross there.

**Helicity modulus.** Twisting the boundary conditions of an O($n$) model by an
angle $\phi$ in the (0,1) spin plane along direction $a$ costs a free energy
$\tfrac12\Upsilon\,\phi^2 L^{d-2}$. Expanding to second order gives
$$ \Upsilon_a = \frac{1}{N}\Big\langle \sum_{\langle ij\rangle_a}
(S_i^0S_j^0 + S_i^1S_j^1)\Big\rangle
- \frac{\beta}{N}\Big\langle\Big(\sum_{\langle ij\rangle_a}(S_i^0S_j^1 - S_i^1S_j^0)\Big)^2\Big\rangle ,$$
which for XY spins is the familiar
$\frac1N\langle\sum\cos(\theta_i-\theta_j)\rangle - \frac{\beta}{N}\langle(\sum\sin(\theta_j-\theta_i))^2\rangle$.
The kernels accumulate both bond sums per axis at every measurement.

---

## 2. Markov-chain Monte Carlo

We want expectations under the Boltzmann distribution
$\pi(x) = e^{-\beta H(x)}/Z$. A Markov chain with transition kernel $P(x\to y)$
samples $\pi$ if it is irreducible, aperiodic and $\pi$ is stationary,
$\sum_x \pi(x)P(x\to y) = \pi(y)$. A sufficient condition is **detailed balance**
$$\pi(x)P(x\to y) = \pi(y)P(y\to x). \tag{2.1}$$
Composing kernels that each leave $\pi$ invariant leaves it invariant, which
is why update schedules such as `"metropolis+3*overrelax+wolff"` are valid even
though the sequential composition does not satisfy (2.1) itself.

### 2.1 Metropolis and heat bath

Metropolis: propose $y$ with a symmetric proposal and accept with probability
$\min(1, e^{-\beta\Delta E})$. For a single Ising spin
$\Delta E = 2s_i(h_i + h)$ with local field $h_i = \sum_{j\sim i}s_j$; the
kernel pre-tabulates the $2(2z+1)$ possible acceptance probabilities.
Sequential (typewriter) sweeps violate detailed balance but satisfy global
balance, being a product of single-site kernels that each satisfy (2.1).

Heat bath (Glauber): resample one variable from its conditional distribution.
Ising: $P(s_i = +1) = [1 + e^{-2\beta(h_i+h)}]^{-1}$. Potts:
$P(s_i = k)\propto e^{\beta n_k}$ with $n_k$ the number of neighbours in state
$k$. Heisenberg: $u = \cos\vartheta$ relative to the local field has density
$\propto e^{\kappa u}$ on $[-1,1]$ with $\kappa = \beta|\mathbf h_i|$, sampled by
inversion as
$$ u = 1 + \kappa^{-1}\ln\!\left(r + (1-r)e^{-2\kappa}\right),\quad r\sim U(0,1),$$
with a uniformly random azimuth. XY: the angle relative to the field is von
Mises distributed, sampled with the rejection algorithm of Best & Fisher
(1979), using the rearrangement
$\rho = 2\kappa\tau/[(\sqrt{1+4\kappa^2}+1)(\tau+\sqrt{2\tau})]$ to avoid
cancellation at small $\kappa$.

For continuous spins the Metropolis proposal is
$\mathbf S' = (\mathbf S + \epsilon\,\mathbf g)/|\mathbf S + \epsilon\,\mathbf g|$ with
isotropic Gaussian $\mathbf g$; by rotational symmetry its density depends only on
$\mathbf S\cdot\mathbf S'$, so it is symmetric. The step $\epsilon$ is tuned towards
50 % acceptance during thermalisation only, keeping the production chain
time-homogeneous.

### 2.2 Over-relaxation

Reflecting a spin about its local field,
$\mathbf S' = 2\frac{(\mathbf S\cdot\mathbf h)}{|\mathbf h|^2}\mathbf h - \mathbf S$, conserves the energy and
is its own inverse, so it satisfies (2.1) trivially. It is not ergodic (it is
microcanonical) but decorrelates quickly when mixed with Metropolis or heat-bath
steps.

### 2.3 Cluster algorithms (Fortuin–Kasteleyn)

The Fortuin–Kasteleyn representation writes the Potts partition function as a
joint model of spins and bond variables $n_{ij}\in\{0,1\}$:
$$ Z = \sum_{\{s\}}\sum_{\{n\}}\prod_{\langle ij\rangle}\left[(1-p)\delta_{n_{ij},0}
+ p\,\delta_{n_{ij},1}\delta_{s_is_j}\right],\qquad p = 1 - e^{-\beta}. \tag{2.2}$$
Summing out $n$ recovers $e^{\beta\delta_{s_is_j}}$ up to a constant. Given the
spins, the bonds are independent: $n_{ij} = 1$ with probability $p$ if
$s_i = s_j$. Given the bonds, every connected cluster carries a single, freely
choosable colour. **Swendsen–Wang** alternates the two conditional updates
(bond activation plus union–find labelling, then a uniformly random new state
per cluster). For the Ising model $p = 1 - e^{-2\beta}$ because the coupling is
$s_is_j$ rather than $\delta_{s_is_j}$.

**Wolff** grows only the cluster containing a random seed and flips it. The
probability of building cluster $C$ from state $x$ contains
$(1-p)^{|\partial C_=|}$, where $\partial C_=$ is the set of boundary bonds
whose outside spin equals the cluster spin. For the reverse move the same
cluster must stop at the boundary bonds that are satisfied *after* the flip.
The ratio of construction probabilities is therefore
$(1-p)^{|\partial C_=| - |\partial C_{\neq}|} = e^{-\beta\Delta E}$, exactly
the Boltzmann ratio, so every proposal is accepted. Potts clusters are
recoloured to a uniformly chosen different state (the choice is symmetric).
The kernel flips members as they are added and marks them with a
cluster-specific tag, so the energy change can be computed from the boundary
bonds in $O(|C|)$ time without clearing arrays.

**O($n$) embedding** (Wolff 1989). Choose a random unit vector $\mathbf r$. The
reflection $R_{\mathbf r}\mathbf S = \mathbf S - 2(\mathbf S\cdot\mathbf r)\mathbf r$ splits every spin into
an Ising variable $\sigma_i = \operatorname{sign}(\mathbf S_i\cdot\mathbf r)$ and a
frozen remainder. The energy restricted to reflections is that of an Ising
model with couplings $J_{ij} = |\mathbf S_i\cdot\mathbf r||\mathbf S_j\cdot\mathbf r|$, so the
Ising cluster rules apply with $p_{ij} = 1 - e^{-2\beta J_{ij}}$ for aligned
$\sigma$. Averaging over $\mathbf r$ restores ergodicity together with local moves.

**Improved estimators.** For Swendsen–Wang in the Ising model,
$\langle M^2\rangle = \langle\sum_C|C|^2\rangle$. For Wolff, whose seed selects
clusters proportionally to size, $\langle M^2\rangle/N = \langle|C|\rangle$.
The kernels record both, and they also convert Wolff time to sweeps
($|C|/N$ per flip).

### 2.4 Dynamic critical behaviour

Near $T_c$ the relaxation time of local algorithms grows like
$\tau\sim\xi^z\to L^z$ with $z\approx 2.17$ (Nightingale & Blöte 1996):
*critical slowing down*. Cluster algorithms update correlated regions
collectively and reduce $z$ to $\approx 0.25$ (or a logarithm). Experiment E04
measures $\tau_\text{int}(L)$ for all four algorithms at the exact $\beta_c$.

---

## 3. Exact solutions

### 3.1 Onsager (infinite square lattice)

With $K = \beta$, $\kappa = 2\sinh 2K/\cosh^2 2K$ and the complete elliptic
integrals $\mathbf K(\kappa)$, $\mathbf E(\kappa)$ of modulus $\kappa$:
$$ -\beta f = \frac{\ln 2}{2} + \frac1{2\pi}\int_0^\pi
\ln\!\Big[\cosh^2 2K + \frac1k\sqrt{1 + k^2 - 2k\cos 2\theta}\Big]d\theta,\qquad k = \sinh^{-2}2K,$$
$$ u = -\coth 2K\Big[1 + \frac2\pi(2\tanh^2 2K - 1)\mathbf K(\kappa)\Big],$$
$$ m_0 = (1 - \sinh^{-4} 2K)^{1/8}\ (T<T_c),\qquad K_c = \tfrac12\ln(1+\sqrt2). $$
The specific heat is obtained by differentiating $u$ analytically, with
$d\mathbf K/d\kappa = \mathbf E/[\kappa(1-\kappa^2)] - \mathbf K/\kappa$ and
$d\kappa/dK = 4(1-\sinh^2 2K)/\cosh^3 2K$. The indeterminate $0\cdot\infty$ at
$K_c$ is resolved by its limit. The implementation is checked against
numerical derivatives of $f$ and $u$, and against Kaufman's formula at
$L = 400$ to $10^{-9}$.

### 3.2 Kaufman (finite torus)

For an $m\times n$ torus (Kaufman 1949, in Beale's 1996 notation)
$$ Z = \tfrac12(2\sinh 2K)^{mn/2}\sum_{i=1}^4 Z_i,\qquad
Z_{1,2} = \prod_{r=0}^{n-1}\!\left\{\begin{matrix}2\cosh\\2\sinh\end{matrix}\right\}\!\big(\tfrac m2\gamma_{2r+1}\big),\quad
Z_{3,4} = \prod_{r=0}^{n-1}\!\left\{\begin{matrix}2\cosh\\2\sinh\end{matrix}\right\}\!\big(\tfrac m2\gamma_{2r}\big),$$
with $\cosh\gamma_l = \cosh 2K\coth 2K - \cos(\pi l/n)$ for $l\ge1$ and the
**signed** $\gamma_0 = 2K + \ln\tanh K$, which changes sign at $K_c$ and makes
$Z_4$ change sign there. We evaluate $\ln Z$ in log space and differentiate
twice *analytically*:

* $\cosh\gamma - 1 = (\sinh 2K - 1)^2/\sinh 2K + 2\sin^2(\pi l/2n)$ is computed
  without cancellation; $\gamma' = c'/\sinh\gamma$ and
  $\gamma'' = (c'' - \cosh\gamma\,\gamma'^2)/\sinh\gamma$ with
  $c = \cosh^2 2K/\sinh 2K$.
* For a product $\prod g_r$ of $2\cosh x_r$ or $2\sinh x_r$ factors the log-derivatives
  are $a_r = t_rx_r'$ and $g_r''/g_r = x_r'^2 + t_rx_r''$ with $t = \tanh$ or
  $\coth$, and $Z''/Z = \sum_r g_r''/g_r + \sum_{r\ne s}a_ra_s$.
* The factor $2\sinh(m\gamma_0/2)$ vanishes at $K_c$. It is handled through
  $\hat w_4 = w_4\coth(m\gamma_0/2) = 2\cosh(m\gamma_0/2)R/Z$, so no $0\cdot\infty$
  and no $\coth^2 - \sinh^{-2}$ cancellation ever occurs.

This gives $\ln Z$, $u$ and $c$ to machine precision for any lattice size
(tested against brute-force enumeration on seven small tori, including odd
sizes and rectangles). Two facts serve as further checks. Below $T_c$,
$f_L = f_\infty - T\ln 2/N$ up to exponentially small terms (the two-fold
degenerate ordered state). And
$c(K_c, L) = \tfrac{8K_c^2}{\pi}\ln L + \dots$ (Ferdinand & Fisher 1969).

### 3.3 Exact density of states by complex-temperature evaluation

On the torus the bond energy takes the values $E_j = -2N + 4j$, so
$$ Z(K) = e^{2NK}P(x),\qquad P(x) = \sum_{j=0}^N g_jx^j,\qquad x = e^{-4K}, \tag{3.1}$$
with the integer density of states $g_j$. If $P$ can be evaluated at the $S$
points $x_k = e^{2\pi i(k+\frac12)/S}$ on the unit circle ($S\ge N+1$, a power
of two), an inverse FFT returns
$g_j = S^{-1}\sum_k P(x_k)\,e^{-2\pi i j(k+\frac12)/S}$ exactly. The half-step
twist avoids $x = 1$, i.e. $K = 0$, where Kaufman's formula is singular.

This requires Kaufman's formula at **complex** $K = -\tfrac14\ln x$, where the
branch of every $\gamma_l = \operatorname{arccosh}(\cdot)$ is ambiguous. For
even $m, n$ all ambiguities cancel:

* the factors with $l$ and $2n-l$ are equal and combine to
  $(2\cosh\frac{m\gamma_l}{2})^2 = 2 + 2T_m(\cosh\gamma_l)$ and
  $(2\sinh\frac{m\gamma_l}{2})^2 = 2T_m(\cosh\gamma_l) - 2$, polynomials in
  $\cosh\gamma_l$ (Chebyshev $T_m$). Numerically we use $w = (y + \sqrt{y^2-1})^{m/2}$
  with $y = \cosh\gamma_l$ and form $(w \pm w^{-1})^2$, which are invariant under
  $w\to w^{-1}$;
* $\gamma_0$ enters only through $e^{\gamma_0} = e^{2K}\tanh K$, which is single-valued;
* the unpaired $\gamma_n$ has $\cosh\gamma_n = c+1$ and the analytic
  $\sinh\gamma_n = \cosh 2K\,(1 + \sinh 2K)/\sinh 2K$, which fixes the sign of
  $2\sinh(m\gamma_n/2)$.

With about $0.30N$ significant digits (the $g_j$ reach $2^N/\sqrt N$),
`mpmath` evaluates $P(x_k)$ and the FFT recovers integers to $10^{-20}$. Every
result is verified against $\sum_j g_j = 2^N$, the symmetry $g_j = g_{N-j}$ and
$g_0 = 2$, $g_1 = 0$, $g_2 = 2N$, and reproduces Kaufman's real-$K$
thermodynamics to $10^{-13}$. $L = 32$ takes 5 s; $L = 64$
($g_\text{max}\approx 10^{1231}$) 3.6 minutes on one core. Beale (1996) obtained
the same numbers by symbolic polynomial expansion.

### 3.4 Fisher zeros

In $u = e^{-2K}$ we have $Z\propto P(u^2)$, a polynomial of degree $2N$ in $u$
with zeros symmetric under $u\to -u$ and complex conjugation. As
$L\to\infty$ they condense on the circles $|u\pm1| = \sqrt2$, which cross the
positive axis at $u_c = \sqrt2 - 1 = e^{-2K_c}$. The zero closest to $u_c$
approaches it as
$$ |u_1(L) - u_c| \sim L^{-1/\nu}, $$
because the zeros of $Z$ are zeros of the scaling function $Z(tL^{1/\nu})$. The
complete zero set is obtained with Durand–Kerner iteration (`mpmath.polyroots`,
feasible to $L\approx12$). The leading zero alone is refined by Newton's method
on the analytically continued $P$, for any even $L$. The zeros are dense
(spacing $\sim 0.1/L$ near $u_c$), so Newton needs starting points inside
basins of width $\sim 0.1/L$. We extrapolate $u_1(L) = u_c + a/L + b/L^2$
through the two previous sizes, which gives such starts and makes $L = 512$
cost about a second.

### 3.5 Other exact results

* **Potts, square lattice.** Self-duality gives $\beta_c = \ln(1+\sqrt q)$ and
  $e(\beta_c) = -(1 + 1/\sqrt q)$. For $q>4$ the transition is first order,
  with latent heat (Baxter 1973)
  $$ \Delta e = 2\left(1 + \tfrac1{\sqrt q}\right)\tanh\tfrac\theta2\prod_{n\ge1}\tanh^2 n\theta,\qquad
  2\cosh\theta = \sqrt q, $$
  i.e. $\Delta e = 0.696049$ for $q = 10$.
* **O($n$) rings.** The transfer operator $e^{\beta\mathbf S\cdot\mathbf S'}$ is
  diagonalised by Fourier modes ($n = 2$: $\lambda_m = 2\pi I_m(\beta)$) and by
  spherical harmonics ($n = 3$, Funk–Hecke: $\lambda_l = 4\pi i_l(\beta)$ with
  degeneracy $2l+1$). Hence $Z_\text{ring} = \sum_l d_l\lambda_l^N$ exactly, and the
  derivatives $I_m'$, $i_l'$, $i_l''$ give $\langle E\rangle$ and $c$. These test
  every continuous-spin kernel at any $N$.
* **Brute force.** Joint densities of states $g(E, M)$ by enumeration, used
  for $N\le 25$ Ising and $q^N\lesssim 3\times10^7$ Potts states, and for
  spin-glass samples.

---

## 4. Flat-histogram and extended-ensemble methods

### 4.1 Wang–Landau with the $1/t$ refinement

Wang–Landau (2001) walks in energy space with acceptance
$\min[1, g(E)/g(E')]$ while adding $\ln f$ to $\ln g(E)$ at every step. When the
visit histogram $H(E)$ is flat ($\min H\ge 0.8\,\bar H$ over the levels seen so
far), $H$ is reset and $\ln f\to\ln f/2$. The estimate converges to $g$ up to a
constant, but the error *saturates*: halving $\ln f$ geometrically freezes the
estimate before the statistical noise has averaged out (Belardinelli &
Pereyra 2007). Their fix: once $\ln f < N_\text{levels}/t$ ($t$ = number of trial
moves), set $\ln f(t) = N_\text{levels}/t$ from then on. The error then decays as
$t^{-1/2}$, which experiment E05 reproduces against the exact $g(E)$.

Energy windows are supported. A move leaving the window is rejected and the
current level updated, as for any rejection (Schulz *et al.* 2003). The
normalisation uses $\sum_E g(E) = q^N$ on the full range, or fixes the
lowest level of a window.

### 4.2 Multicanonical sampling

With weights $w(E) = 1/\hat g(E)$ the energy histogram is flat, and
configurations of any energy, in particular the mixed-phase states of a
first-order transition, are sampled equally often (Berg & Neuhaus 1992).
Canonical averages follow by reweighting each sample with
$e^{-\beta E}\hat g(E)$:
$$ \langle O\rangle_\beta = \frac{\sum_t O_t\,e^{-\beta E_t}\hat g(E_t)}{\sum_t e^{-\beta E_t}\hat g(E_t)}.$$
Any smooth inaccuracy of $\hat g$ cancels exactly: it only changes how often
each level is visited, not the reweighted averages.

**First-order observables.** At the transition $P_\beta(e)$ is double-peaked.
The *equal-weight* point $\beta_\text{eqw}(L)$ approaches $\beta_c$ with
$O(L^{-d})$ corrections (Borgs–Kotecký). The phase-conditional mean
energies at $\beta_\text{eqw}$ approach the coexisting energies $e_o$, $e_d$.
The valley between the peaks, suppressed by two interfaces of length $L$ on
the torus, gives the interface tension:
$2\sigma_{od} = \lim_L \frac1L\ln\frac{P_\text{max}}{P_\text{min}}$. Peak heights
and the valley depth are read from a lightly smoothed histogram (Gaussian
kernel of width $0.01$ in $e$, well below the peak widths), because single
levels are noisy.

### 4.3 Parallel tempering

$M$ replicas run at inverse temperatures $\beta_1<\dots<\beta_M$. After local
sweeps, neighbouring pairs exchange configurations with probability
$$\min\!\left[1, e^{(\beta_k - \beta_{k+1})(E_k - E_{k+1})}\right], \tag{4.1}$$
which satisfies detailed balance for the product ensemble
$\prod_k\pi_{\beta_k}$. Exchanges permute an index array instead of copying
spins. Each configuration carries a label, "last visited the hottest" or "last
visited the coldest" temperature, which gives round-trip counts. It also gives
the fraction $f(T)$ of up-moving replicas, from which the feedback-optimised
temperature set of Katzgraber, Trebst, Huse & Troyer (2006) follows as the
quantiles of $\eta(T)\propto\sqrt{(df/dT)/\Delta T}$.

### 4.4 Spin-glass observables

With two replica sets $a, b$ at each temperature:
$q = N^{-1}\sum_i s^a_is^b_i$, the link overlap
$q_l = N_b^{-1}\sum_{\langle ij\rangle} s^a_is^a_js^b_is^b_j$, and the Fourier overlap
$q(\mathbf k) = N^{-1}\sum_i s^a_is^b_ie^{i\mathbf k\cdot\mathbf r_i}$ at
$\mathbf k_\text{min} = 2\pi/L$ (averaged over directions). Disorder averages
$[\cdot]$ give
$$ g = \tfrac12\left(3 - \frac{[\langle q^4\rangle]}{[\langle q^2\rangle]^2}\right),\qquad
\chi_{SG}(\mathbf k) = N[\langle|q(\mathbf k)|^2\rangle],\qquad
\xi_L = \frac{1}{2\sin(k_\text{min}/2)}\sqrt{\frac{\chi_{SG}(0)}{\chi_{SG}(\mathbf k_\text{min})} - 1}. $$
Both $g$ and $\xi_L/L$ are dimensionless and cross at $T_c$.

**Equilibration identity.** For Gaussian couplings, integration by parts
($[Jf(J)] = [f'(J)]$) applied to each bond gives
$[J_{ij}\langle s_is_j\rangle] = \beta(1 - [\langle s_is_j\rangle^2])$. Hence
$$ [\langle e\rangle] = -\frac z2\,\frac{1 - [\langle q_l\rangle]}{T}, \tag{4.2}$$
exactly in equilibrium (Katzgraber, Palassini & Young 2001). Measured as a
function of log-binned simulation time, the two sides approach each other
from opposite directions and agree once equilibrium is reached. The identity
holds only after the disorder average; single samples scatter by $O(1)$ per
bond, so it is tested with sample-to-sample standard errors.

---

## 5. Statistical analysis of correlated data

### 5.1 Integrated autocorrelation time

For a stationary series with autocovariance $\Gamma(t)$, the variance of the
sample mean is
$$\sigma^2_{\bar A} = \frac{2\tau_\text{int}}{N}\Gamma(0),\qquad
\tau_\text{int} = \frac12 + \sum_{t=1}^\infty\rho(t),\quad\rho(t) = \Gamma(t)/\Gamma(0). \tag{5.1}$$
$\Gamma(t)$ is computed by FFT with zero padding to $\ge 2N$ (no wrap-around)
and normalised by $1/(N-t)$. Summing the noisy tail of $\rho(t)$ would give an
estimator with infinite variance, so the sum is truncated at a window $W$.
Madras & Sokal choose the smallest $W\ge c\,\tau_\text{int}(W)$ ($c\approx6$),
with $\operatorname{var}\hat\tau\approx 2(2W+1)\tau^2/N$.

### 5.2 The Γ-method

Wolff's Γ-method (2004) extends this to derived quantities
$F = f(\bar A_1,\dots,\bar A_n)$, for example
$c = \beta^2(\langle E^2\rangle - \langle E\rangle^2)/N$ or the Binder cumulant.
The fluctuations are projected on the gradient of $f$,
$\delta_F(i) = \sum_\alpha f_\alpha(a^i_\alpha - \bar a_\alpha)$, with symmetric
finite-difference derivatives of step $\sqrt{\Gamma_{\alpha\alpha}(0)/N}$, and
$\Gamma_F(t)$ is summed over independent replicas. The window is the first $W$ with
$$ g(W) = e^{-W/\tau_W} - \frac{\tau_W}{\sqrt{WN}} < 0,\qquad
\tau_W = \frac{S}{\ln\frac{2\tau_\text{int}(W)+1}{2\tau_\text{int}(W)-1}}, $$
which minimises the sum of the truncation bias $\sim e^{-W/\tau}$ and the
statistical error $\sim\sqrt{W/N}$ for an exponential decay ($S\approx1.5$).
The bias $-C_F/N$ of every $\Gamma_F(t)$, caused by subtracting the sample
mean, is corrected before computing
$\sigma_F = \sqrt{C_F/N}$, $C_F = \Gamma_F(0) + 2\sum_{t=1}^W\Gamma_F(t)$. The
implementation agrees with an independent transcription of Wolff's `UWerr` to
$10^{-15}$.

### 5.3 Blocking

Flyvbjerg–Petersen blocking repeatedly averages neighbouring pairs; the naive
error of the blocked series rises and plateaus once blocks are longer than
$\tau$. The plateau is detected automatically with Jonsson's (2018) test:
with lag-1 autocovariances $\gamma_k$ and variances $\sigma_k^2$ at level $k$,
$$ M_j = \sum_{k\ge j} n_k\frac{\left[(n_k-1)\sigma^2_k/n_k^2 + \gamma_k\right]^2}{\sigma^4_k}
\sim\chi^2_{d-j} $$
under the hypothesis that level $j$ is already uncorrelated; the first level
passing at 99 % confidence is used.

### 5.4 Jackknife and bootstrap

Non-linear estimators (reweighted curves, crossing points, peak positions,
data-collapse parameters) get delete-one-block jackknife errors. Block $b$ is
removed, the *entire* analysis is repeated, and
$$\hat\theta_\text{JK} = n\hat\theta - (n-1)\bar\theta_{(\cdot)},\qquad
\sigma^2_\text{JK} = \frac{n-1}{n}\sum_b(\hat\theta_{(b)} - \bar\theta_{(\cdot)})^2 .$$
With blocks much longer than $\tau$ the blocks are nearly independent. For
estimators combining independent data sets (two lattice sizes in a Binder
crossing) the jackknife variances of the data sets add. Disorder averages
use the bootstrap over samples. For time series we also provide moving-block
and stationary (Politis–Romano) bootstraps, and percentile, basic, normal
and BCa intervals.

---

## 6. Histogram reweighting

### 6.1 Single histogram

Samples from $\pi_{\beta_0}$ estimate
$\langle O\rangle_\beta = \sum_tO_te^{-(\beta-\beta_0)E_t}/\sum_te^{-(\beta-\beta_0)E_t}$
(Ferrenberg & Swendsen 1988). The effective sample size
$(\sum w)^2/\sum w^2$ decays like $e^{-(\Delta\beta\,\sigma_E)^2}$ in the
Gaussian approximation, so the useful range is
$|\Delta\beta|\lesssim 1/\sigma_E\sim L^{-d/2}$. Temperature derivatives follow
without numerical differentiation:
$\partial_\beta\langle O\rangle = \langle O\rangle\langle E\rangle - \langle OE\rangle$.

### 6.2 Multiple histograms (MBAR)

Runs at $\beta_1,\dots,\beta_K$ with $N_k$ samples are combined optimally by
the self-consistent free energies $f_k = -\ln Z(\beta_k)$ (Ferrenberg &
Swendsen 1989; Shirts & Chodera 2008). These are the minimiser of the
convex function
$$ F(\mathbf f) = \sum_n\ln\sum_k N_ke^{f_k - \beta_kE_n} - \sum_kN_kf_k , $$
which we minimise by Newton's method with exact Hessian
$H_{ij} = \sum_n(\delta_{ij}W_{ni} - W_{ni}W_{nj})$ and a backtracking line
search, in the gauge $f_1 = 0$. The starting point chains single-histogram
estimates between neighbouring temperatures. Correlated runs enter with
effective sizes $N_k/g_k$, $g_k = 2\tau_k$. For discrete energies the sums run
over distinct energy values, which makes solving fast enough to repeat inside
every jackknife replicate.

---

## 7. Finite-size scaling

Near a continuous transition the singular part of the free energy of a
system of size $L$ obeys
$f_s(t, L) = L^{-d}\mathcal F(tL^{1/\nu}, uL^{-\omega})$ with $t = \beta - \beta_c$
and an irrelevant field $u$. Differentiating gives the estimators used in E03
and E08:

| quantity | scaling at $\beta_c$ |
|---|---|
| $U_4(\beta, L)$ | $U^* + a\,tL^{1/\nu} + b\,L^{-\omega}+\dots$ |
| $\partial_\beta U_4$, $\partial_\beta\ln\langle m^2\rangle$ | $\propto L^{1/\nu}(1 + cL^{-\omega})$ |
| $\chi_0 = \beta N\langle m^2\rangle$ | $\propto L^{\gamma/\nu}$ |
| $\langle|m|\rangle$ | $\propto L^{-\beta/\nu}$ |
| $\chi'_\text{max}(L)$ and its position | $\propto L^{\gamma/\nu}$, $\beta_\text{max}(L) - \beta_c\propto L^{-1/\nu}$ |
| $c_\text{max}(L)$ | $\propto L^{\alpha/\nu}$; 2D Ising: $A\ln L$ |
| Binder crossing $\beta^*(L, bL)$ | $\beta_c + a L^{-1/\nu-\omega}$ |

For the 2D Ising model the leading corrections are analytic ($\omega = 2$) and
all exponents are known exactly: $\nu = 1$, $\gamma/\nu = 7/4$,
$\beta/\nu = 1/8$, $U^* = 0.61069$ on the square torus.

**Data collapse.** If $O(\beta, L) = L^\kappa\mathcal G(tL^{1/\nu})$, then the
curves $O L^{-\kappa}$ plotted against $tL^{1/\nu}$ coincide for the correct
$(\beta_c, 1/\nu, \kappa)$. The quality of a collapse is measured by the
statistic of Houdayer & Hartmann (2004). For every point of every curve, a
weighted straight line is fitted through the two bracketing points of each
*other* curve. Then
$$ S = \frac1{\mathcal N}\sum\frac{(y - Y)^2}{\Delta y^2 + \Delta Y^2}, $$
where $Y$ and $\Delta Y$ are the line's prediction and its standard error. $S\approx1$
for a collapse consistent with the error bars. We minimise $S$ with
Nelder–Mead, and the errors come from redoing the collapse on every jackknife
replicate of the reweighted curves.

**Berezinskii–Kosterlitz–Thouless.** The XY model has no symmetry breaking; its
correlation length diverges as $\xi\sim e^{b/\sqrt{T - T_\text{BKT}}}$, so
there is no power law to fit. The renormalisation group instead predicts a
universal jump of the helicity modulus,
$\Upsilon(T_\text{BKT}^-) = 2T_\text{BKT}/\pi$ (Nelson & Kosterlitz 1977), and
logarithmic finite-size corrections. At $T_\text{BKT}$,
$\Upsilon(L) = \frac{2T}{\pi}\left[1 + \frac{1}{2\ln L + C}\right]$ (Weber &
Minnhagen 1988), so the temperature where this one-parameter form fits best
estimates $T_\text{BKT}$. The crossings $T^*(L)$ of $\Upsilon(T, L)$ with $2T/\pi$
drift as $T^*(L) = T_\text{BKT} + a/(\ln L)^2$.

---

## References

* R. J. Baxter, *J. Phys. C* **6**, L445 (1973) — Potts latent heat.
* P. D. Beale, *Phys. Rev. Lett.* **76**, 78 (1996) — exact distribution of energies.
* R. E. Belardinelli, V. D. Pereyra, *Phys. Rev. E* **75**, 046701 (2007) — 1/t Wang–Landau.
* B. A. Berg, T. Neuhaus, *Phys. Rev. Lett.* **68**, 9 (1992) — multicanonical ensemble.
* D. J. Best, N. I. Fisher, *Appl. Statist.* **28**, 152 (1979) — von Mises sampling.
* C. Borgs, R. Kotecký, *J. Stat. Phys.* **61**, 79 (1990) — first-order finite-size scaling.
* A. E. Ferdinand, M. E. Fisher, *Phys. Rev.* **185**, 832 (1969) — finite-size specific heat.
* A. M. Ferrenberg, R. H. Swendsen, *Phys. Rev. Lett.* **61**, 2635 (1988); **63**, 1195 (1989).
* M. E. Fisher, in *Lectures in Theoretical Physics* VIIC (1965) — partition-function zeros.
* H. Flyvbjerg, H. G. Petersen, *J. Chem. Phys.* **91**, 461 (1989) — blocking.
* J. Houdayer, A. K. Hartmann, *Phys. Rev. B* **70**, 014418 (2004) — collapse quality.
* M. Jonsson, *Phys. Rev. E* **98**, 043304 (2018) — automated blocking.
* B. Kaufman, *Phys. Rev.* **76**, 1232 (1949) — finite-lattice partition function.
* H. G. Katzgraber, M. Palassini, A. P. Young, *Phys. Rev. B* **63**, 184422 (2001).
* H. G. Katzgraber, S. Trebst, D. A. Huse, M. Troyer, *J. Stat. Mech.* P03018 (2006).
* N. Madras, A. D. Sokal, *J. Stat. Phys.* **50**, 109 (1988) — automatic windowing.
* D. R. Nelson, J. M. Kosterlitz, *Phys. Rev. Lett.* **39**, 1201 (1977) — universal jump.
* M. P. Nightingale, H. W. J. Blöte, *Phys. Rev. Lett.* **76**, 4548 (1996) — Metropolis z.
* L. Onsager, *Phys. Rev.* **65**, 117 (1944).
* D. N. Politis, J. P. Romano, *J. Amer. Statist. Assoc.* **89**, 1303 (1994) — stationary bootstrap.
* B. J. Schulz, K. Binder, M. Müller, D. P. Landau, *Phys. Rev. E* **67**, 067102 (2003).
* M. R. Shirts, J. D. Chodera, *J. Chem. Phys.* **129**, 124105 (2008) — MBAR.
* R. H. Swendsen, J.-S. Wang, *Phys. Rev. Lett.* **58**, 86 (1987).
* F. Wang, D. P. Landau, *Phys. Rev. Lett.* **86**, 2050 (2001).
* H. Weber, P. Minnhagen, *Phys. Rev. B* **37**, 5986 (1988).
* U. Wolff, *Phys. Rev. Lett.* **62**, 361 (1989) — single-cluster algorithm.
* U. Wolff, *Comput. Phys. Commun.* **156**, 143 (2004) — Γ-method.
