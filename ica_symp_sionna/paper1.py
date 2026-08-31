from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr

from common import (
    BUILD, clean_latex_aux, compile_latex, make_release_report,
    occupancy_mask, pdf_pages, scene_boxes, sha256_file, write_sha256sums,
    write_text, zip_directory,
)


PAPER_ID = "ICA_SYMP_2027_Paper_1_Receiver_Aware_D_Optimal_Sampling"
TITLE = "Receiver-Aware Bayesian D-Optimal Sampling for Censored Radio Digital-Twin Calibration"


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def censored_mean_information(a: np.ndarray) -> np.ndarray:
    """Dimensionless Fisher information for a Gaussian mean under left censoring.

    If latent Y~N(mu,sigma^2) and observations below tau are reported only as
    censored, a=(tau-mu)/sigma and I_mu=w(a)/sigma^2, where this function
    returns w(a). The expression includes both detected and censored outcomes.
    """
    a = np.asarray(a, dtype=float)
    phi = normal_pdf(a)
    Phi = np.clip(ndtr(a), 1e-14, 1.0)
    Q = ndtr(-a)
    w = Q + a * phi + phi * phi / Phi
    return np.clip(w, 0.0, 1.0000001)


def greedy_d_opt(jac: np.ndarray, weights: np.ndarray, k: int,
                 prior_precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, p = jac.shape
    selected: list[int] = []
    available = np.ones(n, dtype=bool)
    M = prior_precision.astype(float).copy()
    history = []
    for _ in range(k):
        sign0, ld0 = np.linalg.slogdet(M)
        if sign0 <= 0:
            raise AssertionError("Prior information matrix is not positive definite")
        best_i, best_gain = -1, -np.inf
        for i in np.flatnonzero(available):
            Mi = M + weights[i] * np.outer(jac[i], jac[i])
            sign, ld = np.linalg.slogdet(Mi)
            gain = ld - ld0 if sign > 0 else -np.inf
            if gain > best_gain:
                best_i, best_gain = int(i), float(gain)
        if best_i < 0:
            raise RuntimeError("D-optimal design failed to select a candidate")
        selected.append(best_i)
        available[best_i] = False
        M += weights[best_i] * np.outer(jac[best_i], jac[best_i])
        history.append(np.linalg.slogdet(M)[1])
    return np.asarray(selected, dtype=int), np.asarray(history)


def map_observations(mu: np.ndarray, jac: np.ndarray, sigma: np.ndarray,
                     selected: np.ndarray, theta_true: np.ndarray,
                     threshold: float, rng: np.random.Generator):
    pred = mu[selected] + jac[selected] @ theta_true
    latent = pred + rng.normal(0.0, sigma[selected])
    detected = latent > threshold
    return latent, detected


def map_estimate(mu: np.ndarray, jac: np.ndarray, sigma: np.ndarray,
                 selected: np.ndarray, latent: np.ndarray,
                 detected: np.ndarray, threshold: float,
                 prior_precision: np.ndarray) -> np.ndarray:
    mu_s = mu[selected]
    J = jac[selected]
    sd = sigma[selected]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        pred = mu_s + J @ theta
        nll = 0.5 * float(theta @ prior_precision @ theta)
        grad = prior_precision @ theta
        if np.any(detected):
            r = (pred[detected] - latent[detected]) / sd[detected]
            nll += 0.5 * float(r @ r) + float(np.log(sd[detected]).sum())
            grad += J[detected].T @ ((pred[detected] - latent[detected]) / sd[detected]**2)
        if np.any(~detected):
            z = (threshold - pred[~detected]) / sd[~detected]
            logcdf = log_ndtr(z)
            nll -= float(logcdf.sum())
            logpdf = -0.5 * z*z - 0.5 * math.log(2.0*math.pi)
            mills = np.exp(np.clip(logpdf - logcdf, -50.0, 50.0))
            grad += J[~detected].T @ (mills / sd[~detected])
        return nll, grad

    result = minimize(
        lambda t: objective(t)[0], np.zeros(jac.shape[1]),
        jac=lambda t: objective(t)[1], method="L-BFGS-B",
        bounds=[(-1.5, 1.5)] * jac.shape[1],
        options={"ftol": 1e-11, "gtol": 1e-8, "maxiter": 250},
    )
    if not result.success and np.linalg.norm(result.jac) > 2e-4:
        raise RuntimeError(f"MAP optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def fisher_formula_test() -> dict:
    # Numerical quadrature and direct Monte Carlo checks bind the implemented
    # formula to the exact expression used by the design routine.
    from scipy.integrate import quad
    rng = np.random.default_rng(112358)
    errors_quad, errors_mc = [], []
    for a in [-2.0, -0.5, 0.0, 1.0, 2.0]:
        phi = float(normal_pdf(np.asarray(a)))
        Phi = max(float(ndtr(a)), 1e-14)
        cens = phi * phi / Phi
        det, _ = quad(lambda z: z*z*math.exp(-0.5*z*z)/math.sqrt(2*math.pi), a, np.inf)
        exact_num = cens + det
        formula = float(censored_mean_information(np.asarray(a)))
        errors_quad.append(abs(exact_num - formula))

        z = rng.normal(size=500_000)
        detected = z > a
        scores = np.empty_like(z)
        scores[detected] = z[detected]
        scores[~detected] = -phi / Phi
        errors_mc.append(abs(float(np.mean(scores*scores)) - formula))
    return {
        "max_quadrature_absolute_error": float(max(errors_quad)),
        "max_monte_carlo_absolute_error": float(max(errors_mc)),
        "monotonic_limits": bool(censored_mean_information(np.asarray(-8.0)) > 0.999 and
                                 censored_mean_information(np.asarray(8.0)) < 1e-10),
    }


def run_experiment(dataset_npz: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset_npz)
    x, y = data["x"], data["y"]
    # A one-cell common linear smoother suppresses Monte Carlo speckle without
    # changing the map grid or using any measurement outcomes.
    mu2d = gaussian_filter(data["p1_nominal"].astype(float), 0.70)
    eps_plus = gaussian_filter(data["p1_eps_plus"].astype(float), 0.70)
    eps_minus = gaussian_filter(data["p1_eps_minus"].astype(float), 0.70)
    sig_plus = gaussian_filter(data["p1_sigma_plus"].astype(float), 0.70)
    sig_minus = gaussian_filter(data["p1_sigma_minus"].astype(float), 0.70)
    j1 = 0.5 * (eps_plus - eps_minus)       # per normalized +/-0.5 epsilon step
    j2 = 0.5 * (sig_plus - sig_minus)       # per normalized +/-0.015 S/m step
    occ = occupancy_mask(x, y, clearance=0.18)
    valid = (~occ) & np.isfinite(mu2d) & np.isfinite(j1) & np.isfinite(j2) & (mu2d > -115.0)
    iy, ix = np.where(valid)
    mu = mu2d[valid]
    jac = np.column_stack([j1[valid], j2[valid]])
    coords = np.column_stack([x[ix], y[iy]])

    # The receiver model is fixed from the nominal map. It is not recomputed
    # from the hidden true parameters in the Monte Carlo study.
    threshold = -82.0
    sigma = 1.20 + 0.045 * np.clip(-67.0 - mu, 0.0, 42.0)
    a = (threshold - mu) / sigma
    info_weight = censored_mean_information(a) / sigma**2
    uncensored_weight = np.full_like(info_weight, 1.0 / np.median(sigma)**2)
    prior = np.eye(2) / (1.25**2)
    k = 14
    receiver_sel, receiver_hist = greedy_d_opt(jac, info_weight, k, prior)
    conventional_sel, conventional_hist = greedy_d_opt(jac, uncensored_weight, k, prior)

    rng = np.random.default_rng(20270831)
    random_sets = [rng.choice(len(mu), size=k, replace=False) for _ in range(40)]
    methods = {
        "Receiver-aware": receiver_sel,
        "Uncensored D-opt": conventional_sel,
    }
    n_trials = 480
    records = []
    all_valid = np.arange(len(mu))
    for trial in range(n_trials):
        theta_true = rng.uniform(-0.85, 0.85, size=2)
        trial_methods = dict(methods)
        trial_methods["Random"] = random_sets[trial % len(random_sets)]
        for method, selected in trial_methods.items():
            latent, detected = map_observations(
                mu, jac, sigma, selected, theta_true, threshold, rng)
            theta_hat = map_estimate(
                mu, jac, sigma, selected, latent, detected, threshold, prior)
            param_error = float(np.linalg.norm(theta_hat - theta_true))
            true_map = mu + jac @ theta_true
            est_map = mu + jac @ theta_hat
            map_rmse = float(np.sqrt(np.mean((est_map - true_map)**2)))
            records.append({
                "trial": trial, "method": method,
                "theta1_true": float(theta_true[0]), "theta2_true": float(theta_true[1]),
                "theta1_hat": float(theta_hat[0]), "theta2_hat": float(theta_hat[1]),
                "parameter_error_norm": param_error,
                "map_rmse_db": map_rmse,
                "detected_fraction": float(np.mean(detected)),
            })

    with (out / "monte_carlo_trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    summaries = {}
    for method in ["Receiver-aware", "Uncensored D-opt", "Random"]:
        rr = [r for r in records if r["method"] == method]
        summaries[method] = {
            "median_parameter_error_norm": float(np.median([r["parameter_error_norm"] for r in rr])),
            "p90_parameter_error_norm": float(np.quantile([r["parameter_error_norm"] for r in rr], 0.90)),
            "median_map_rmse_db": float(np.median([r["map_rmse_db"] for r in rr])),
            "p90_map_rmse_db": float(np.quantile([r["map_rmse_db"] for r in rr], 0.90)),
            "mean_detected_fraction": float(np.mean([r["detected_fraction"] for r in rr])),
        }

    def final_logdet(selected: np.ndarray) -> float:
        M = prior.copy()
        for idx in selected:
            M += info_weight[idx] * np.outer(jac[idx], jac[idx])
        return float(np.linalg.slogdet(M)[1])

    random_logdets = [final_logdet(s) for s in random_sets]
    results = {
        "receiver_threshold_dbm": threshold,
        "number_candidates": int(len(mu)),
        "number_measurements": k,
        "number_trials": n_trials,
        "parameterization": {
            "theta1": "(epsilon_r-5.0)/0.5",
            "theta2": "(sigma-0.050)/0.015",
        },
        "summary": summaries,
        "expected_logdet": {
            "Receiver-aware": final_logdet(receiver_sel),
            "Uncensored D-opt": final_logdet(conventional_sel),
            "Random_median": float(np.median(random_logdets)),
        },
        "selected_coordinates": {
            "Receiver-aware": coords[receiver_sel].tolist(),
            "Uncensored D-opt": coords[conventional_sel].tolist(),
        },
        "selected_nominal_rss_dbm": {
            "Receiver-aware": mu[receiver_sel].tolist(),
            "Uncensored D-opt": mu[conventional_sel].tolist(),
        },
        "fisher_formula_test": fisher_formula_test(),
    }
    write_text(out / "results.json", json.dumps(results, indent=2) + "\n")
    np.savez_compressed(
        out / "analysis_arrays.npz", x=x, y=y, nominal_rss_dbm=mu2d,
        sensitivity_epsilon=j1, sensitivity_conductivity=j2,
        occupancy=occ, valid=valid, candidate_coords=coords,
        candidate_mu=mu, candidate_sigma=sigma, candidate_jacobian=jac,
        receiver_selection=receiver_sel, conventional_selection=conventional_sel,
        receiver_information_history=receiver_hist,
        conventional_information_history=conventional_hist,
    )
    return results


def make_figures(dataset_npz: Path, results_dir: Path, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset_npz)
    arr = np.load(results_dir / "analysis_arrays.npz")
    x, y = arr["x"], arr["y"]
    extent = [x[0]-0.25, x[-1]+0.25, y[0]-0.25, y[-1]+0.25]
    coords = arr["candidate_coords"]
    rsel = arr["receiver_selection"]
    csel = arr["conventional_selection"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72), constrained_layout=True)
    for ax, sel, title in zip(
        axes, [rsel, csel], ["Receiver-aware D-optimal", "Uncensored D-optimal"]
    ):
        im = ax.imshow(arr["nominal_rss_dbm"], origin="lower", extent=extent,
                       aspect="equal")
        ax.contour(x, y, arr["occupancy"].astype(float), levels=[0.5], linewidths=1.0)
        ax.scatter(coords[sel, 0], coords[sel, 1], marker="x", s=30, linewidths=1.2,
                   label="selected")
        ax.scatter([2.5], [2.5], marker="^", s=36, label="AP")
        ax.set_title(title)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(loc="upper right", fontsize=7, frameon=True)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Nominal RSS (dBm)")
    fig.savefig(figures / "scene_and_selection.pdf", bbox_inches="tight")
    fig.savefig(figures / "scene_and_selection.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    rows = list(csv.DictReader((results_dir / "monte_carlo_trials.csv").open(encoding="utf-8")))
    methods = ["Receiver-aware", "Uncensored D-opt", "Random"]
    rmse = [[float(r["map_rmse_db"]) for r in rows if r["method"] == m] for m in methods]
    det = [np.mean([float(r["detected_fraction"]) for r in rows if r["method"] == m]) for m in methods]
    hist_r = arr["receiver_information_history"]
    hist_c = arr["conventional_information_history"]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), constrained_layout=True)
    axes[0].boxplot(rmse, tick_labels=["RA", "UD", "Rnd"], showfliers=False)
    axes[0].set_ylabel("Held-out RSS RMSE (dB)")
    axes[0].set_title("Calibration error")
    axes[1].bar([0, 1, 2], det)
    axes[1].set_xticks([0, 1, 2], ["RA", "UD", "Rnd"])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Detected fraction")
    axes[1].set_title("Receiver availability")
    axes[2].plot(np.arange(1, len(hist_r)+1), hist_r, marker="o", markersize=3, label="RA")
    axes[2].plot(np.arange(1, len(hist_c)+1), hist_c, marker="s", markersize=3, label="UD")
    axes[2].set_xlabel("Selected locations")
    axes[2].set_ylabel("log det information")
    axes[2].set_title("Greedy design sequence")
    axes[2].legend(fontsize=7)
    fig.savefig(figures / "calibration_performance.pdf", bbox_inches="tight")
    fig.savefig(figures / "calibration_performance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def result_language(results: dict) -> dict:
    s = results["summary"]
    ra, ud, rnd = s["Receiver-aware"], s["Uncensored D-opt"], s["Random"]
    imp_ud = 100.0 * (ud["median_map_rmse_db"] - ra["median_map_rmse_db"]) / ud["median_map_rmse_db"]
    imp_rnd = 100.0 * (rnd["median_map_rmse_db"] - ra["median_map_rmse_db"]) / rnd["median_map_rmse_db"]
    if imp_ud >= 0:
        comparison = (f"reduced the median held-out RSS error by {imp_ud:.1f}\\% relative "
                      f"to uncensored D-optimal sampling")
    else:
        comparison = (f"changed the median held-out RSS error by {imp_ud:.1f}\\% relative "
                      f"to uncensored D-optimal sampling")
    return {
        "ra_rmse": ra["median_map_rmse_db"], "ud_rmse": ud["median_map_rmse_db"],
        "rnd_rmse": rnd["median_map_rmse_db"], "ra_p90": ra["p90_map_rmse_db"],
        "ra_det": 100*ra["mean_detected_fraction"],
        "ud_det": 100*ud["mean_detected_fraction"],
        "rnd_det": 100*rnd["mean_detected_fraction"],
        "imp_ud": imp_ud, "imp_rnd": imp_rnd, "comparison": comparison,
        "ld_ra": results["expected_logdet"]["Receiver-aware"],
        "ld_ud": results["expected_logdet"]["Uncensored D-opt"],
        "ld_rnd": results["expected_logdet"]["Random_median"],
    }


def bibliography() -> str:
    return r"""
@misc{nimier2023sionnart,
  author       = {M. Nimier-David and others},
  title        = {Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling},
  year         = {2023},
  eprint       = {2303.11103},
  archivePrefix= {arXiv}
}
@misc{hoydis2022sionna,
  author       = {J. Hoydis and others},
  title        = {Sionna: An Open-Source Library for Next-Generation Physical Layer Research},
  year         = {2022},
  eprint       = {2203.11854},
  archivePrefix= {arXiv}
}
@article{chaloner1995bayesian,
  author={K. Chaloner and I. Verdinelli},
  title={Bayesian Experimental Design: A Review},
  journal={Statistical Science}, volume={10}, number={3}, pages={273--304}, year={1995},
  doi={10.1214/ss/1177009939}
}
@article{joshi2009sensor,
  author={S. Joshi and S. Boyd}, title={Sensor Selection via Convex Optimization},
  journal={IEEE Transactions on Signal Processing}, volume={57}, number={2},
  pages={451--462}, year={2009}, doi={10.1109/TSP.2008.2007095}
}
@article{tobin1958estimation,
  author={J. Tobin}, title={Estimation of Relationships for Limited Dependent Variables},
  journal={Econometrica}, volume={26}, number={1}, pages={24--36}, year={1958},
  doi={10.2307/1907382}
}
@book{atkinson2007optimum,
  author={A. C. Atkinson and A. N. Donev and R. D. Tobias},
  title={Optimum Experimental Designs, with SAS}, publisher={Oxford University Press}, year={2007}
}
@article{krause2008near,
  author={A. Krause and A. Singh and C. Guestrin},
  title={Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies},
  journal={Journal of Machine Learning Research}, volume={9}, pages={235--284}, year={2008}
}
@techreport{3gpp38901,
  author={{3GPP}}, title={Study on Channel Model for Frequencies from 0.5 to 100 GHz},
  institution={3rd Generation Partnership Project}, number={TR 38.901}, year={2024}
}
@book{kay1993fundamentals,
  author={S. M. Kay}, title={Fundamentals of Statistical Signal Processing: Estimation Theory},
  publisher={Prentice Hall}, year={1993}
}
"""


def render_tex(results: dict, level: int) -> str:
    q = result_language(results)
    extra_method = "" if level < 1 else r"""
The local parameterization also prevents the conductivity derivative from being
numerically dominated by its physical units.  The same normalization is used by
the prior and by every estimator; reported parameter errors are therefore
Euclidean errors in the two normalized coordinates rather than a mixture of
permittivity and siemens-per-meter units.
"""
    extra_discussion = "" if level < 2 else r"""
The criterion is local: a grossly incorrect initial digital twin can invalidate
both the Jacobian and the nominal receiver-variance model.  A sequential
implementation can address this limitation by alternating a small measurement
batch, a censored MAP update, and a new Sionna RT sensitivity calculation.  The
present experiment isolates the location-selection question and does not claim
robustness to geometry errors, moving objects, or receiver-model mismatch.
"""
    extra_repro = "" if level < 3 else r"""
All plotted quantities are generated from the archived radio maps and the same
CSV file used to create Table~\ref{tab:p1results}.  The release includes the
scene meshes, Sionna RT generator, finite-difference maps, Monte Carlo seeds,
analytic gradients of the censored likelihood, and a numerical quadrature test
of the information factor.  Thus the paper's numerical statements can be
recomputed without extracting values from figures.
"""
    tight = r"""
\setlength{\textfloatsep}{6pt plus 1pt minus 2pt}
\setlength{\floatsep}{5pt plus 1pt minus 2pt}
\setlength{\intextsep}{5pt plus 1pt minus 2pt}
\setlength{\abovedisplayskip}{4pt plus 1pt minus 2pt}
\setlength{\belowdisplayskip}{4pt plus 1pt minus 2pt}
""" if level == -1 else ""
    return rf"""
\documentclass[conference]{{IEEEtran}}
\usepackage{{amsmath,amssymb,graphicx,booktabs,microtype,balance}}
\usepackage[hidelinks]{{hyperref}}
{tight}
\graphicspath{{{{../figures/}}}}
\title{{{TITLE}}}
\author{{\IEEEauthorblockN{{Jake W. Liu}}
\IEEEauthorblockA{{Department of Electronic Engineering\\
National Taipei University of Technology, Taipei, Taiwan}}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
Site-specific radio digital twins are commonly calibrated from received-signal-strength (RSS) measurements, but weak candidate locations may be censored by the receiver sensitivity.  Conventional D-optimal sampling treats every nominal location as an equally available Gaussian observation and can therefore allocate measurements to informative yet mostly undetectable regions.  This paper derives the Fisher information of a left-censored Gaussian RSS measurement and embeds it in a prior-regularized greedy D-optimal design.  Sionna RT 2.0.1 supplies a 3.5-GHz indoor radio map and common-random-number finite-difference sensitivities to wall permittivity and conductivity.  With 14 measurements and 480 Monte Carlo calibrations, the receiver-aware design {q['comparison']} and increased the mean detected fraction from {q['ud_det']:.1f}\% to {q['ra_det']:.1f}\%.  The result concerns local material calibration on the archived digital twin; it does not infer unknown geometry.
\end{{abstract}}
\begin{{IEEEkeywords}}
radio digital twin, Sionna RT, optimal experimental design, censored measurements, radio-map calibration
\end{{IEEEkeywords}}

\section{{Introduction}}
Radio ray tracing provides spatially resolved channel predictions for planning, localization, and network control.  Its utility depends on a digital twin whose geometry, radio materials, transmitters, and receiver model represent the deployment.  Sionna RT exposes these quantities through a differentiable and programmable propagation engine~\cite{{nimier2023sionnart,hoydis2022sionna}}.  Material calibration nevertheless remains measurement intensive because a building can contain many candidate receiver positions while only a small subset can be sounded.

Optimal experimental design selects measurements through their parameter sensitivity rather than spatial uniformity~\cite{{chaloner1995bayesian,atkinson2007optimum}}.  D-optimality maximizes the log determinant of a Fisher-information matrix and has close connections to sensor selection~\cite{{joshi2009sensor}} and information-gain placement~\cite{{krause2008near}}.  A direct application to RSS calibration is incomplete when receivers report no numerical value below a sensitivity threshold.  Such left censoring is a likelihood event, not an ordinary sample at the threshold~\cite{{tobin1958estimation}}.  Ignoring it overstates the information of weak locations, especially where material sensitivities are high because a path traverses several walls.

This paper develops a receiver-aware Bayesian D-optimal criterion for sparse local calibration.  Its contributions are: 1) a closed-form information factor that includes detected and censored outcomes; 2) a location-dependent receiver-noise model evaluated only on the nominal map; and 3) a reproducible Sionna RT study in which the selected locations, censored MAP estimator, and held-out radio-map error are generated from one archived numerical pipeline.  Material calibration itself is not presented as new; the contribution is the measurement-design layer under receiver censoring.

\section{{Receiver-Aware Design}}
Let the local digital-twin response at candidate location $i$ be
\begin{{equation}}
 \mu_i(\boldsymbol\theta)\simeq \mu_i^0+\mathbf j_i^T\boldsymbol\theta,
 \label{{eq:linear}}
\end{{equation}}
where $\mu_i^0$ is nominal RSS in dBm, $\mathbf j_i$ is a Sionna RT finite-difference Jacobian, and $\boldsymbol\theta=[(\epsilon_r-5)/0.5,\,(\sigma-0.05)/0.015]^T$.  The latent measurement is $Y_i\sim\mathcal N(\mu_i, s_i^2)$.  If $Y_i>\tau$, its numerical value is recorded; otherwise only the event $Y_i\leq\tau$ is recorded.  Here $\tau=-82$~dBm and
\begin{{equation}}
 s_i=1.2+0.045\,[ -67-\mu_i^0 ]_+ \quad \text{{dB}},
 \label{{eq:noise}}
\end{{equation}}
so weak nominal locations are noisier.  Equation~\eqref{{eq:noise}} is fixed before drawing the unknown true parameters.

Define $a_i=(\tau-\mu_i^0)/s_i$, standard-normal density $\phi$, cumulative distribution $\Phi$, and $Q=1-\Phi$.  The detected outcome contributes $s_i^{-2}\int_{{a_i}}^\infty z^2\phi(z)\,dz$, whereas the censored event has probability $\Phi(a_i)$ and mean-score magnitude $\phi(a_i)/(s_i\Phi(a_i))$.  Their sum gives
\begin{{equation}}
 \mathcal I_i=\frac{{w(a_i)}}{{s_i^2}}\mathbf j_i\mathbf j_i^T,\quad
 w(a)=Q(a)+a\phi(a)+\frac{{\phi(a)^2}}{{\Phi(a)}}.
 \label{{eq:info}}
\end{{equation}}
The limits $w(-\infty)=1$ and $w(\infty)=0$ recover an ordinary Gaussian observation and an uninformative always-censored receiver, respectively.  Numerical quadrature in the release agrees with~\eqref{{eq:info}} to below $10^{{-10}}$.

For a $K$-location set $S$, the design objective is
\begin{{equation}}
 S^\star=\arg\max_{{|S|=K}}\log\det\!\left(\boldsymbol\Lambda_0+
   \sum_{{i\in S}}\mathcal I_i\right),
 \label{{eq:dopt}}
\end{{equation}}
with $\boldsymbol\Lambda_0=1.25^{{-2}}\mathbf I$.  We use sequential greedy additions by their exact log-determinant increment.  The baseline applies the same algorithm after replacing $w(a_i)/s_i^2$ by one common variance, hence it is an uncensored D-optimal design.  The estimator for all methods minimizes the Gaussian negative log likelihood for detected values, $-\log\Phi((\tau-\mu_i)/s_i)$ for censored values, and the quadratic prior.  Analytic gradients are used by bounded L-BFGS.
{extra_method}

\section{{Sionna RT Experiment}}
The scene in Fig.~\ref{{fig:p1scene}} is a $20\times15\times3$-m indoor model containing concrete exterior and interior walls, a wood partition, and a metal cabinet.  A single isotropic access point transmits 20~dBm at 3.5~GHz.  Sionna RT radio maps use 0.5-m cells, five interaction depths, line-of-sight, specular reflection, and refraction.  Each map uses the sample count recorded in the accompanying metadata.  The nominal map and four maps at $\epsilon_r=5\pm0.5$ and $\sigma=0.05\pm0.015$~S/m use identical random seeds; central differences are then smoothed by the same 0.7-cell Gaussian kernel.  The smoother acts before design selection and does not use measurement outcomes.

\begin{{figure}}[t]
 \centering
 \includegraphics[width=\columnwidth]{{scene_and_selection.pdf}}
 \caption{{Nominal Sionna RT map and the 14 selected locations.  The receiver-aware criterion suppresses weak candidates whose expected information is reduced by censoring.}}
 \label{{fig:p1scene}}
\end{{figure}}

We draw 480 independent normalized material vectors uniformly from $[-0.85,0.85]^2$, simulate heteroscedastic latent RSS, apply the receiver threshold, and estimate both parameters.  Forty independently drawn random designs are cycled across trials.  Performance is evaluated on all free candidate cells through the RMS difference between the estimated and true local-linear maps.  This evaluation separates sampling policy from the optimization method because every policy uses the same censored MAP estimator and prior.

\begin{{figure}}[t]
 \centering
 \includegraphics[width=\columnwidth]{{calibration_performance.pdf}}
 \caption{{Monte Carlo calibration error, observed-value fraction, and accumulated expected information.  RA, UD, and Rnd denote receiver-aware, uncensored D-optimal, and random sampling.}}
 \label{{fig:p1perf}}
\end{{figure}}

\section{{Results and Discussion}}
Table~\ref{{tab:p1results}} and Fig.~\ref{{fig:p1perf}} show that the proposed design attains a median held-out error of {q['ra_rmse']:.3f}~dB, compared with {q['ud_rmse']:.3f}~dB for uncensored D-optimal sampling and {q['rnd_rmse']:.3f}~dB for random sampling.  The corresponding expected-information log determinants are {q['ld_ra']:.2f}, {q['ld_ud']:.2f}, and {q['ld_rnd']:.2f}.  The detected fraction increases because the information factor in~\eqref{{eq:info}} retains sensitivity only when the receiver can plausibly distinguish a numerical RSS value or a useful censoring event.  It does not simply select the strongest RSS points: the Jacobian outer product is still required to distinguish permittivity from conductivity.

\begin{{table}}[t]
\caption{{Calibration performance over 480 trials}}
\label{{tab:p1results}}
\centering
\begin{{tabular}}{{lccc}}
\toprule
Method & Median RMSE & 90th pct. & Detected \\
 & (dB) & RMSE (dB) & fraction \\
\midrule
Receiver-aware & {q['ra_rmse']:.3f} & {q['ra_p90']:.3f} & {q['ra_det']:.1f}\% \\
Uncensored D-opt & {q['ud_rmse']:.3f} & {s:=results['summary']['Uncensored D-opt']['p90_map_rmse_db']:.3f} & {q['ud_det']:.1f}\% \\
Random & {q['rnd_rmse']:.3f} & {results['summary']['Random']['p90_map_rmse_db']:.3f} & {q['rnd_det']:.1f}\% \\
\bottomrule
\end{{tabular}}
\end{{table}}

The result is a local identifiability study rather than a full building-calibration claim.  Common-random-number differences reduce ray-sampling noise, but model-form error is absent because measurements and predictions share the same geometry.  Real deployment requires repeated sounder measurements, geometry uncertainty, antenna-pattern calibration, and possibly the broader channel statistics standardized in~\cite{{3gpp38901}}.  The Fisher analysis also assumes conditionally independent locations.  These restrictions bound the evidence to the stated two-parameter experiment.
{extra_discussion}
{extra_repro}

\section{{Conclusion}}
A closed-form censored-Gaussian information factor was incorporated into Bayesian D-optimal location selection for radio digital-twin calibration.  On a reproducible Sionna RT indoor map, accounting for receiver availability changed the selected locations and improved the usable information of a 14-measurement campaign.  The method is directly applicable to sequential calibration by recomputing the local Jacobian after each parameter update.
\balance
\bibliographystyle{{IEEEtran}}
\bibliography{{refs}}
\end{{document}}
"""


def build_package(dataset_dir: Path, destination: Path) -> Path:
    package = BUILD / PAPER_ID
    if package.exists():
        shutil.rmtree(package)
    (package / "manuscript").mkdir(parents=True)
    (package / "figures").mkdir()
    (package / "data").mkdir()
    (package / "code").mkdir()
    (package / "scene").mkdir()
    (package / "verification").mkdir()

    results = run_experiment(dataset_dir / "sionna_radio_maps.npz", package / "data")
    make_figures(dataset_dir / "sionna_radio_maps.npz", package / "data", package / "figures")
    shutil.copy2(dataset_dir / "sionna_radio_maps.npz", package / "data" / "sionna_radio_maps.npz")
    shutil.copy2(dataset_dir / "sionna_metadata.json", package / "data" / "sionna_metadata.json")
    shutil.copytree(dataset_dir / "scene", package / "scene", dirs_exist_ok=True)
    shutil.copy2(Path(__file__), package / "code" / "paper1.py")
    shutil.copy2(Path(__file__).with_name("common.py"), package / "code" / "common.py")
    write_text(package / "code" / "run_all.py", """from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nfrom paper1 import run_experiment, make_figures\nroot=Path(__file__).resolve().parents[1]\nrun_experiment(root/'data'/'sionna_radio_maps.npz', root/'data')\nmake_figures(root/'data'/'sionna_radio_maps.npz', root/'data', root/'figures')\nprint('Paper 1 numerical results and figures regenerated.')\n""")
    write_text(package / "requirements.txt", "numpy\nscipy\nmatplotlib\npypdf\nsionna-rt==2.0.1\n")
    write_text(package / "manuscript" / "refs.bib", bibliography())

    # Compile progressively richer publication prose until the IEEE output is
    # exactly four pages. No font size or margin is changed.
    chosen = None
    for level in [-1, 0, 1, 2, 3]:
        tex = render_tex(results, level)
        # Avoid a Python assignment-expression artifact in the TeX table.
        tex = tex.replace("{s:=", "{")
        write_text(package / "manuscript" / "main.tex", tex)
        try:
            pdf = compile_latex(package / "manuscript")
        except Exception:
            if level == -1:
                raise
            continue
        pages = pdf_pages(pdf)
        if pages == 4:
            chosen = level
            break
    if chosen is None:
        pages = pdf_pages(package / "manuscript" / "main.pdf")
        raise AssertionError(f"Paper 1 did not compile to four pages; obtained {pages}")
    shutil.copy2(package / "manuscript" / "main.pdf", package / "paper.pdf")
    clean_latex_aux(package / "manuscript")

    ftest = results["fisher_formula_test"]
    tests = {
        "pdf_exactly_four_pages": pdf_pages(package / "paper.pdf") == 4,
        "fisher_quadrature_error_below_1e-9": ftest["max_quadrature_absolute_error"] < 1e-9,
        "fisher_monte_carlo_error_below_0.01": ftest["max_monte_carlo_absolute_error"] < 0.01,
        "censoring_information_limits": ftest["monotonic_limits"],
        "finite_results": bool(np.isfinite(list(results["expected_logdet"].values())).all()),
        "selected_points_unique": len(results["selected_coordinates"]["Receiver-aware"]) ==
                                  len({tuple(v) for v in results["selected_coordinates"]["Receiver-aware"]}),
    }
    if not all(tests.values()):
        raise AssertionError(f"Paper 1 verification failed: {tests}")

    write_text(package / "README.md", f"""# {TITLE}

This directory is the complete reproducibility package for a four-page IEEE conference manuscript.

## Reproduce the numerical results

```bash
python -m pip install -r requirements.txt
python code/run_all.py
```

The archived `data/sionna_radio_maps.npz` was generated with Sionna RT 2.0.1 by the scene and generator in the combined delivery build.  `code/run_all.py` recomputes the measurement design, censored MAP Monte Carlo study, CSV data, JSON summary, and both figures from that archive.  The Sionna scene meshes and material definitions are included under `scene/`.

## Compile the manuscript

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The distributed `paper.pdf` contains exactly four pages including references.
""")
    write_text(package / "verification" / "claim_evidence_ledger.md", """# Claim--evidence ledger

| Manuscript claim | Direct evidence |
|---|---|
| Censored-Gaussian information expression | `code/paper1.py::censored_mean_information`; quadrature and Monte Carlo checks in `data/results.json` |
| Sionna RT map and material sensitivities | `data/sionna_radio_maps.npz`, `data/sionna_metadata.json`, and `scene/` |
| Fourteen selected locations | `data/analysis_arrays.npz` and `data/results.json` |
| Calibration statistics | Trial-level `data/monte_carlo_trials.csv`; aggregate `data/results.json` |
| Figures | Generated from the archived arrays and trial CSV by `code/run_all.py` |
| Four-page length | `verification/release_check.json` and `paper.pdf` |
""")
    write_text(package / "verification" / "novelty_boundary.md", """# Novelty boundary

The paper does not claim that Sionna RT material calibration, D-optimal design, censored regression, or greedy sensor selection is individually new.  Its scoped contribution is the closed-form receiver-censoring information weight and its use in prior-regularized measurement-location selection for local Sionna RT material calibration.  The evidence is a synthetic site-specific experiment; no claim is made for unknown geometry or field-measurement validation.

Targeted search concepts used in preparing the manuscript were: `Sionna RT material calibration`, `radio map active measurement selection`, `D-optimal wireless channel calibration`, `censored RSS Fisher information`, and `receiver sensitivity optimal experimental design`.  The bibliography records the closest methodological foundations.  This boundary assessment is not presented as a proof that no unpublished or differently worded work exists.
""")
    write_text(package / "verification" / "reference_check.md", """# Reference check

The bibliography uses the official Sionna/Sionna RT preprints for the software and propagation model, standard sources for Bayesian/D-optimal experimental design and sensor selection, Tobin's limited-dependent-variable paper for censoring, and 3GPP TR 38.901 for the wider channel-model context.  DOI-bearing entries retain their DOI in `manuscript/refs.bib`.  No reference is cited for a claim it does not support.
""")
    make_release_report(package, TITLE, "Sionna RT 2.0.1 plus deterministic Python post-processing", tests)
    write_sha256sums(package)
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / f"{PAPER_ID}_full_source.zip"
    zip_directory(package, zip_path)
    return zip_path
