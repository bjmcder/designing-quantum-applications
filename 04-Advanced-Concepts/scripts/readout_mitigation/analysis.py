"""Analysis: build the confusion matrix, correct <Z>, and compare the methods.

From the counts of execution.py this module computes four estimates of <Z>
at every RY angle:

unmitigated   (n0 - n1) / N from the plain runs. Biased by a z + b.
inverted      Apply the inverse of the confusion matrix, which is the same as
              (z_measured - b) / a, with p01 and p10 from the two calibration
              states.
twirled       The mean over the randomizations of the flipped-back answers.
              Biased by the factor a only.
rescaled      The twirled value divided by a, where a is the mean twirled
              <Z> of the calibration state |0> (its true value is 1).

Error bars are one standard error. For a plain run it comes from the binomial
spread of the counts. For a twirled run it is the spread of the
randomizations divided by sqrt(R), which also covers the finite number of
coin flips. The error of the calibration is added in quadrature with the
first-order propagation formula for z = (z_m - b) / a:

    var(z) = [var(z_m) + var(b) + z^2 var(a) + 2 z cov(a, b)] / a^2.

Prints a table of the bias and draws these figures (colors are the same in
every panel):

- the confusion matrix
- the error rate of a true 0 and a true 1, with and without the twirl
- measured against ideal <Z>
- the four estimates and their residuals (estimate minus ideal)
- the spread of the twirled estimate against the number of randomizations
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from ops import confusion_matrix, ideal_z

# Color-blind-safe colors, one per method (used in every panel).
COLORS = {
    "unmitigated": "#D55E00",
    "inverted": "#0072B2",
    "rescaled": "#009E73",
    "twirled": "#CC79A7",
}
LABELS = {
    "unmitigated": "unmitigated",
    "inverted": "matrix inversion",
    "twirled": "twirled, not rescaled",
    "rescaled": "twirled and rescaled",
}


def z_from_counts(n0, n1) -> tuple[np.ndarray, np.ndarray]:
    """<Z> = P(0) - P(1) and its standard error from counts.

    The error uses the estimate (n0 + 1) / (N + 2) of P(0), which stays
    above zero when all shots agree.

    Args:
        n0: Number of 0s (scalar or array).
        n1: Number of 1s.

    Returns:
        (z, standard error).
    """
    n0, n1 = np.asarray(n0, float), np.asarray(n1, float)
    total = n0 + n1
    p = (n0 + 1) / (total + 2)
    return (n0 - n1) / total, 2 * np.sqrt(p * (1 - p) / total)


def twirled_z(pairs) -> tuple[float, float]:
    """Mean <Z> over the randomizations and its standard error.

    Args:
        pairs: One (n0, n1) per randomization, in the bit labels of the ideal circuit.

    Returns:
        (mean, standard deviation of the randomizations / sqrt(R)).
    """
    arr = np.asarray(pairs, float)
    z = (arr[:, 0] - arr[:, 1]) / arr.sum(axis=1)
    return float(z.mean()), float(z.std(ddof=1) / np.sqrt(len(z)))


def calibrate(data: dict) -> dict:
    """The read-out error from the calibration counts.

    Args:
        data: Output of ``execution.run``.

    Returns:
        p01 and p10 with standard errors, the confusion matrix, the
        attenuation a and offset b with their errors and covariance, and
        the twirled attenuation read from the twirled |0>.
    """
    n00, n01 = data["calibration"]["0"]  # prepared 0: read 0, read 1
    n10, n11 = data["calibration"]["1"]
    p01 = n01 / (n00 + n01)
    p10 = n10 / (n10 + n11)
    v01 = p01 * (1 - p01) / (n00 + n01)
    v10 = p10 * (1 - p10) / (n10 + n11)
    a_twirl, a_twirl_err = twirled_z(data["calibration_twirled"]["0"])
    return {
        "p01": p01,
        "p10": p10,
        "p01_err": float(np.sqrt(v01)),
        "p10_err": float(np.sqrt(v10)),
        "matrix": confusion_matrix(p01, p10),
        "a": 1 - p01 - p10,
        "b": p10 - p01,
        "var_a": v01 + v10,  # a = 1 - p01 - p10
        "var_b": v01 + v10,  # b = p10 - p01
        "cov_ab": v01 - v10,
        "a_twirl": a_twirl,
        "a_twirl_err": a_twirl_err,
    }


def invert(counts, cal: dict) -> tuple[float, float, float, float]:
    """Correct one plain run by the inverse of the confusion matrix.

    Args:
        counts: (n0, n1) of the run.
        cal: Output of ``calibrate``.

    Returns:
        (corrected <Z>, its standard error, measured <Z>, measured error).
    """
    n0, n1 = counts
    measured = np.array([n0, n1]) / (n0 + n1)
    fixed = np.linalg.solve(cal["matrix"], measured)  # C^-1 times the data
    z = float(fixed[0] - fixed[1])
    zm, zm_err = z_from_counts(n0, n1)
    var = (
        zm_err**2 + cal["var_b"] + z**2 * cal["var_a"] + 2 * z * cal["cov_ab"]
    ) / cal["a"] ** 2
    return z, float(np.sqrt(var)), float(zm), float(zm_err)


def estimates(data: dict) -> dict:
    """The four estimates of <Z> at every angle.

    Args:
        data: Output of ``execution.run``.

    Returns:
        ``cal`` (from ``calibrate``), ``ideal``, and for each method in
        ``COLORS`` the arrays ``z`` and ``err``.
    """
    cal = calibrate(data)
    theta = np.array(data["angles"])
    out = {k: {"z": [], "err": []} for k in COLORS}
    for raw, pairs in zip(data["raw"], data["twirled"]):
        z, err, zm, zm_err = invert(raw, cal)
        out["unmitigated"]["z"].append(zm)
        out["unmitigated"]["err"].append(zm_err)
        out["inverted"]["z"].append(z)
        out["inverted"]["err"].append(err)
        zt, zt_err = twirled_z(pairs)
        out["twirled"]["z"].append(zt)
        out["twirled"]["err"].append(zt_err)
        a, a_err = cal["a_twirl"], cal["a_twirl_err"]
        zr = zt / a
        out["rescaled"]["z"].append(zr)
        out["rescaled"]["err"].append(np.sqrt(zt_err**2 + zr**2 * a_err**2) / a)
    for k, series in out.items():
        out[k] = {name: np.array(v) for name, v in series.items()}
    return {"cal": cal, "theta": theta, "ideal": ideal_z(theta), **out}


def print_calibration(data: dict, res: dict) -> None:
    """Print the read-out error and the twirled attenuation.

    Args:
        data: Output of ``execution.run``.
        res: Output of ``estimates``.
    """
    cal, truth = res["cal"], data["truth"]
    built = lambda key: f"{truth[key]:>10.4f}" if truth else f"{'-':>10}"
    print(f"{'':<34} {'measured':>18} {'built in':>10}")
    rows = [
        ("p01 (read 1 when the bit is 0)", cal["p01"], cal["p01_err"], "p01"),
        ("p10 (read 0 when the bit is 1)", cal["p10"], cal["p10_err"], "p10"),
        ("a = 1 - p01 - p10", cal["a"], np.sqrt(cal["var_a"]), "a"),
        ("b = p10 - p01", cal["b"], np.sqrt(cal["var_b"]), "b"),
        ("a from the twirled |0>", cal["a_twirl"], cal["a_twirl_err"], "a"),
    ]
    for label, value, err, key in rows:
        print(f"{label:<34} {value:>9.4f} +/- {err:<6.4f} {built(key)}")


def bias_table(res: dict) -> dict:
    """Residual bias of each method over the sweep.

    Args:
        res: Output of ``estimates``.

    Returns:
        For each method the mean residual, the largest absolute residual,
        and the mean error bar.
    """
    table = {}
    for name in COLORS:
        resid = res[name]["z"] - res["ideal"]
        table[name] = {
            "mean": float(resid.mean()),
            "max": float(np.abs(resid).max()),
            "error": float(res[name]["err"].mean()),
        }
    return table


def print_bias(res: dict) -> None:
    """Print the residual bias and error bar of each method.

    Args:
        res: Output of ``estimates``.
    """
    print(f"{'method':<24} {'mean bias':>10} {'max |bias|':>11} {'error bar':>10}")
    for name, row in bias_table(res).items():
        print(
            f"{LABELS[name]:<24} {row['mean']:>+10.3f} {row['max']:>11.3f} "
            f"{row['error']:>10.3f}"
        )


def error_rates(data: dict) -> dict:
    """Fraction of wrong answers for a true 0 and a true 1, with and without twirl.

    Args:
        data: Output of ``execution.run``.

    Returns:
        ``plain`` and ``twirled`` lists of (rate, standard error) for the
        true bits 0 and 1.
    """
    out = {"plain": [], "twirled": []}
    for bit in (0, 1):
        n = np.array(data["calibration"][str(bit)], float)
        pairs = np.array(data["calibration_twirled"][str(bit)], float).sum(axis=0)
        for key, counts in (("plain", n), ("twirled", pairs)):
            rate = counts[1 - bit] / counts.sum()
            out[key].append((rate, float(np.sqrt(rate * (1 - rate) / counts.sum()))))
    return out


def analyze(data: dict, out: Path) -> dict:
    """Print the tables and save the comparison figure.

    Args:
        data: Output of ``execution.run``.
        out: Path of the PNG to write.

    Returns:
        The output of ``estimates``.
    """
    res = estimates(data)
    print_calibration(data, res)
    print()
    print_bias(res)
    if data.get("estimator"):
        ev = np.array(data["estimator"]["evs"])
        resid = ev - res["ideal"]
        print(
            f"{'IBM Estimator (TREX)':<24} {resid.mean():>+10.3f} "
            f"{np.abs(resid).max():>11.3f} "
            f"{np.mean(data['estimator']['stds']):>10.3f}"
        )
    fig = plot_compare(res)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")
    return res


# ------------------------------------------------------------------ figures


def plot_confusion(cal: dict):
    """The confusion matrix as a grid of numbers.

    Args:
        cal: Output of ``calibrate``.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    matrix = cal["matrix"]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0.5 else "black",
            )
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_yticks([0, 1], ["0", "1"])
    ax.set_xlabel("prepared bit")
    ax.set_ylabel("read bit")
    ax.set_title("Confusion matrix")
    return fig


def plot_error_rates(data: dict):
    """Error rate of a true 0 and a true 1, with and without the twirl.

    Args:
        data: Output of ``execution.run``.

    Returns:
        The figure.
    """
    rates = error_rates(data)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    x = np.arange(2)
    for k, (key, label, color) in enumerate(
        [
            ("plain", "no twirl", COLORS["unmitigated"]),
            ("twirled", "twirl", COLORS["rescaled"]),
        ]
    ):
        vals = np.array(rates[key]) * [100, 100]
        ax.bar(
            x + (k - 0.5) * 0.36,
            vals[:, 0],
            0.36,
            yerr=vals[:, 1],
            color=color,
            label=label,
            capsize=3,
        )
    ax.set_xticks(x, ["true 0", "true 1"])
    ax.set_ylabel("error rate (%)")
    ax.set_title("Read-out error rate by true bit")
    ax.legend()
    return fig


def plot_bias(res: dict):
    """Measured against ideal <Z> without any correction.

    Args:
        res: Output of ``estimates``.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([-1, 1], [-1, 1], "k--", lw=1, label="measured = ideal")
    ax.errorbar(
        res["ideal"],
        res["unmitigated"]["z"],
        res["unmitigated"]["err"],
        fmt="o",
        ms=4,
        capsize=2,
        color=COLORS["unmitigated"],
        label="unmitigated",
    )
    ax.set_xlabel("ideal $\\langle Z\\rangle$")
    ax.set_ylabel("measured $\\langle Z\\rangle$")
    ax.set_title("Measured vs ideal expectation value")
    ax.set_aspect("equal")
    ax.legend()
    return fig


def plot_compare(res: dict, methods=("unmitigated", "inverted", "rescaled")):
    """The estimates against the angle, and their residuals.

    Args:
        res: Output of ``estimates``.
        methods: Methods to draw.

    Returns:
        The figure.
    """
    theta = res["theta"]
    step = theta[1] - theta[0]
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.0),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.1},
        sharex=True,
    )
    fine = np.linspace(0, np.pi, 300)
    top.plot(fine, ideal_z(fine), "k-", lw=1, label="ideal")
    bottom.axhline(0, color="black", lw=1)
    for k, name in enumerate(methods):
        shift = (k - (len(methods) - 1) / 2) * step * 0.12  # keep bars apart
        kw = {"fmt": "o", "ms": 3.5, "capsize": 2, "color": COLORS[name]}
        top.errorbar(
            theta + shift, res[name]["z"], res[name]["err"], label=LABELS[name], **kw
        )
        bottom.errorbar(
            theta + shift, res[name]["z"] - res["ideal"], res[name]["err"], **kw
        )
    top.set_ylabel("$\\langle Z\\rangle$")
    top.set_title("Mitigation methods compared")
    top.legend(fontsize=8)
    top.tick_params(labelbottom=False)
    bottom.set_xlabel("RY angle $\\theta$ (rad)")
    bottom.set_ylabel("estimate $-$ ideal")
    return fig


def plot_randomizations(study: dict, a: float, b: float):
    """Spread of the twirled estimate against the number of randomizations.

    The solid line is the model sqrt(1 / N + (b / sqrt(R))^2) / a: shot noise
    plus the leftover offset from an uneven number of flips. The dashed line
    is the shot noise alone.

    Args:
        study: Output of ``execution.randomization_study``.
        a: Attenuation of the read-out.
        b: Offset of the read-out.

    Returns:
        The figure.
    """
    counts = np.array(study["counts"])
    spread = np.array([np.std(study["estimates"][c], ddof=1) for c in counts])
    shots = study["total_shots"]
    shot_noise = 1 / (a * np.sqrt(shots))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(counts, spread, "o", color=COLORS["rescaled"], label="spread of repeats")
    ax.plot(
        counts,
        np.sqrt(shot_noise**2 + (b / a) ** 2 / counts),
        "-",
        color="black",
        lw=1,
        label="shot noise and offset",
    )
    ax.axhline(shot_noise, color="gray", ls="--", lw=1, label="shot noise only")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("randomizations $R$")
    ax.set_ylabel("error bar of $\\langle Z\\rangle$")
    ax.set_title("Error bar vs number of randomizations")
    ax.legend(fontsize=8)
    return fig
