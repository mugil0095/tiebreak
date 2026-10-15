import numpy as np
import pytest
from scipy.stats import norm

from tiebreak.metrics import accuracy
from tiebreak.recommend import recommend
from tiebreak.stats import equivalent_set, paired_bootstrap

pytestmark = pytest.mark.slow

N_SIMS, N_BOOT = 200, 300
RATES: dict[str, float] = {}  # measured rates, printed once at the end of this module's tests
MIN_SHARE = 0.5  # conditional rates are only meaningful if the condition holds in at least this share of reps


@pytest.fixture(scope="module", autouse=True)
def _print_rates(request):
    """Print the measured rates after this module's tests, independent of test order or -s."""
    yield
    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None and RATES:
        tr.write_line("")
        tr.write_line("Measured rates (seeded):")
        for k, v in RATES.items():
            tr.write_line(f"  {v:6.3f}  {k}")


def cond_rate(hits: int, conditioned: int, reps: int) -> float:
    """hits / conditioned, after checking the condition (e.g. 'the best is m0') held often enough."""
    assert conditioned / reps >= MIN_SHARE, f"condition held in only {conditioned}/{reps} reps"
    return hits / conditioned


def simulate(true_acc: list[float], n: int, rng):
    """Correlated candidates: each example has a shared difficulty plus candidate-specific noise.
    Candidate j is correct on an example when its unit-variance latent score exceeds the
    threshold norm.ppf(1 - acc), so its true accuracy is exactly acc.
    Returns (y, preds) as encoded ints; y is all ones and prediction 1 means correct."""
    z = rng.normal(size=n)
    preds = {}
    for j, acc in enumerate(true_acc):
        score = 0.8 * z + 0.6 * rng.normal(size=n)  # 0.8**2 + 0.6**2 = 1, so unit variance
        preds[f"m{j}"] = (score > norm.ppf(1 - acc)).astype(np.int64)
    y = np.ones(n, dtype=np.int64)
    return y, preds


def boot_of(y, preds, rng, n_boot=N_BOOT):
    return paired_bootstrap(y, preds, accuracy, n_boot=n_boot, seed=int(rng.integers(10**9)))


def test_intervals_cover_the_true_accuracy_about_95_percent_of_the_time():
    rng = np.random.default_rng(11)
    accs = [0.90, 0.80]
    truth = accs
    hits = 0
    total = 0
    for _ in range(N_SIMS):
        y, preds = simulate(accs, 200, rng)
        boot = paired_bootstrap(y, preds, accuracy, n_boot=N_BOOT, seed=int(rng.integers(10**9)))
        for j, name in enumerate(boot.names):
            lo, hi = boot.ci[name]
            hits += lo <= truth[j] <= hi
            total += 1
    RATES["CI coverage (nominal 0.95, n=200)"] = hits / total
    assert hits / total >= 0.90  # nominal 95%; allow Monte-Carlo and small-n slack


def test_a_clearly_worse_model_is_almost_never_declared_tied():
    rng = np.random.default_rng(12)
    false_ties = best_m0 = 0
    for _ in range(N_SIMS):
        y, preds = simulate([0.92, 0.72], 200, rng)  # true gap about 0.20, far above the margin
        boot = paired_bootstrap(y, preds, accuracy, n_boot=N_BOOT, seed=int(rng.integers(10**9)))
        if boot.best == "m0":
            best_m0 += 1
            false_ties += equivalent_set(boot, 0.02)[boot.names[1]]
    rate = cond_rate(false_ties, best_m0, N_SIMS)
    RATES["false tie | best is m0, gap 0.20, n=200, margin 0.02"] = rate
    assert rate <= 0.05


def test_a_truly_equal_model_is_usually_declared_tied_with_enough_data():
    rng = np.random.default_rng(13)
    ties = 0
    for _ in range(N_SIMS):
        y, preds = simulate([0.90, 0.90], 800, rng)
        boot = paired_bootstrap(y, preds, accuracy, n_boot=N_BOOT, seed=int(rng.integers(10**9)))
        eq = equivalent_set(boot, 0.03)
        ties += all(eq.values())
    RATES["power: equal models tied, n=800, margin 0.03"] = ties / N_SIMS
    assert ties / N_SIMS >= 0.5  # power check: not required to be perfect, but not useless


def test_small_samples_rarely_claim_a_tie_for_models_that_are_really_different_by_5_points():
    rng = np.random.default_rng(14)
    claims = best_m0 = 0
    for _ in range(N_SIMS):
        y, preds = simulate([0.90, 0.85], 60, rng)
        boot = paired_bootstrap(y, preds, accuracy, n_boot=N_BOOT, seed=int(rng.integers(10**9)))
        if boot.best == "m0":
            best_m0 += 1
            claims += equivalent_set(boot, 0.02)[boot.names[1]]
    rate = cond_rate(claims, best_m0, N_SIMS)
    RATES["false tie | best is m0, gap 0.05, n=60, margin 0.02"] = rate
    assert rate <= 0.10


@pytest.mark.parametrize("n", [100, 300])
def test_false_tie_rate_at_the_margin_boundary_stays_small(n):
    """Best accuracy 0.90; the competitor's errors are a superset of the best's (nested), so the
    true accuracy difference is exactly the margin 0.02. Tying it is a (boundary) false tie."""
    rng = np.random.default_rng(100 + n)
    margin, reps = 0.02, 300
    false_ties = best_m0 = 0
    for _ in range(reps):
        u = rng.random(n)
        y = np.ones(n, dtype=np.int64)
        preds = {"m0": (u >= 0.10).astype(np.int64), "m1": (u >= 0.12).astype(np.int64)}
        boot = boot_of(y, preds, rng, n_boot=400)
        if boot.best == "m0":
            best_m0 += 1
            false_ties += equivalent_set(boot, margin)["m1"]
    rate = cond_rate(false_ties, best_m0, reps)
    RATES[f"false tie | best is m0, at boundary (loss = margin 0.02), n={n}"] = rate
    assert rate <= 0.07


def test_four_equal_candidates_keep_reasonable_power_and_one_worse_one_is_not_tied():
    rng = np.random.default_rng(16)
    reps = 100
    # Known conservatism, characterised rather than fixed: at margin 0.03 the best-of-4 selection gap
    # plus Holm over 3 comparisons leaves almost no power even for a fixed-best bootstrap (about 0.11).
    # The bound below is an UPPER characterisation so a future change that makes this much less
    # conservative (or much more liberal) is noticed and reviewed.
    tight = 0
    for _ in range(reps):
        y, preds = simulate([0.90] * 4, 800, rng)
        tight += all(equivalent_set(boot_of(y, preds, rng), 0.03).values())
    RATES["k=4 equal: all tied, n=800, margin 0.03 (known conservatism)"] = tight / reps
    assert tight / reps <= 0.15
    all_tied = 0
    for _ in range(reps):
        y, preds = simulate([0.90] * 4, 800, rng)
        all_tied += all(equivalent_set(boot_of(y, preds, rng), 0.05).values())
    RATES["k=4 equal: all tied (power), n=800, margin 0.05"] = all_tied / reps
    assert all_tied / reps >= 0.8
    false_ties = not_m3 = 0
    for _ in range(reps):
        y, preds = simulate([0.92, 0.92, 0.92, 0.72], 300, rng)  # m3 is 0.20 worse
        boot = boot_of(y, preds, rng)
        if boot.best != "m3":
            not_m3 += 1
            false_ties += equivalent_set(boot, 0.03)["m3"]
    rate = cond_rate(false_ties, not_m3, reps)
    RATES["k=4: clearly worse candidate tied | best is not m3, n=300"] = rate
    assert rate <= 0.05


def test_recommend_never_crowns_a_truly_worse_cheap_model_and_usually_picks_an_identical_one():
    rng = np.random.default_rng(17)
    reps, margin = 100, 0.03
    cost = {"m0": 10.0, "m1": 1.0}  # m1 is the cheap candidate
    wrong = best_m0 = 0
    for _ in range(reps):
        y, preds = simulate([0.90, 0.75], 300, rng)  # cheap model truly 0.15 worse
        boot = boot_of(y, preds, rng)
        if boot.best == "m0":
            best_m0 += 1
            wrong += recommend(boot, equivalent_set(boot, margin), cost, margin).winner == "m1"
    rate = cond_rate(wrong, best_m0, reps)
    RATES["recommend crowns cheap model 0.15 worse | best is m0, n=300"] = rate
    assert rate <= 0.05
    picked = 0
    for _ in range(reps):
        y, preds = simulate([0.90, 0.90], 800, rng)  # cheap model truly identical
        boot = boot_of(y, preds, rng)
        rec = recommend(boot, equivalent_set(boot, margin), cost, margin)
        picked += rec.winner == "m1"
    RATES["recommend picks identical cheap model, n=800"] = picked / reps
    assert picked / reps >= 0.5
