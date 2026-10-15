import numpy as np
import pytest
from scipy.stats import norm

from tiebreak.metrics import accuracy
from tiebreak.recommend import recommend
from tiebreak.stats import equivalent_set, paired_bootstrap

pytestmark = pytest.mark.slow

N_SIMS, N_BOOT = 200, 300
RATES: dict[str, float] = {}  # measured rates, printed by the last test (run with -s to see them)


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
    return paired_bootstrap(y, preds, accuracy, n_boot=n_boot, seed=int(rng.integers(1e9)))


def test_intervals_cover_the_true_accuracy_about_95_percent_of_the_time():
    rng = np.random.default_rng(11)
    accs = [0.90, 0.80]
    truth = accs
    hits = 0
    total = 0
    for _ in range(N_SIMS):
        y, preds = simulate(accs, 200, rng)
        boot = paired_bootstrap(y, preds, accuracy, n_boot=N_BOOT, seed=int(rng.integers(1e9)))
        for j, name in enumerate(boot.names):
            lo, hi = boot.ci[name]
            hits += lo <= truth[j] <= hi
            total += 1
    RATES["CI coverage (nominal 0.95, n=200)"] = hits / total
    assert hits / total >= 0.90  # nominal 95%; allow Monte-Carlo and small-n slack


def test_a_clearly_worse_model_is_almost_never_declared_tied():
    rng = np.random.default_rng(12)
    false_ties = 0
    for _ in range(N_SIMS):
        y, preds = simulate([0.92, 0.72], 200, rng)  # true gap about 0.20, far above the margin
        boot = paired_bootstrap(y, preds, accuracy, n_boot=N_BOOT, seed=int(rng.integers(1e9)))
        false_ties += equivalent_set(boot, 0.02)[boot.names[1]] if boot.best == "m0" else 0
    RATES["false tie, gap 0.20, n=200, margin 0.02"] = false_ties / N_SIMS
    assert false_ties / N_SIMS <= 0.05


def test_a_truly_equal_model_is_usually_declared_tied_with_enough_data():
    rng = np.random.default_rng(13)
    ties = 0
    for _ in range(N_SIMS):
        y, preds = simulate([0.90, 0.90], 800, rng)
        boot = paired_bootstrap(y, preds, accuracy, n_boot=N_BOOT, seed=int(rng.integers(1e9)))
        eq = equivalent_set(boot, 0.03)
        ties += all(eq.values())
    RATES["power: equal models tied, n=800, margin 0.03"] = ties / N_SIMS
    assert ties / N_SIMS >= 0.5  # power check: not required to be perfect, but not useless


def test_small_samples_rarely_claim_a_tie_for_models_that_are_really_different_by_5_points():
    rng = np.random.default_rng(14)
    claims = 0
    for _ in range(N_SIMS):
        y, preds = simulate([0.90, 0.85], 60, rng)
        boot = paired_bootstrap(y, preds, accuracy, n_boot=N_BOOT, seed=int(rng.integers(1e9)))
        eq = equivalent_set(boot, 0.02)
        claims += eq[boot.names[1]] if boot.best == "m0" else 0
    RATES["false tie, gap 0.05, n=60, margin 0.02"] = claims / N_SIMS
    assert claims / N_SIMS <= 0.10


@pytest.mark.parametrize("n", [100, 300])
def test_false_tie_rate_at_the_margin_boundary_stays_small(n):
    """Best accuracy 0.90; the competitor's errors are a superset of the best's (nested), so the
    true accuracy difference is exactly the margin 0.02. Tying it is a (boundary) false tie."""
    rng = np.random.default_rng(100 + n)
    margin, reps = 0.02, 300
    false_ties = 0
    for _ in range(reps):
        u = rng.random(n)
        y = np.ones(n, dtype=np.int64)
        preds = {"m0": (u >= 0.10).astype(np.int64), "m1": (u >= 0.12).astype(np.int64)}
        boot = boot_of(y, preds, rng, n_boot=400)
        eq = equivalent_set(boot, margin)
        false_ties += eq["m1"] if boot.best == "m0" else 0
    RATES[f"false tie at boundary (loss = margin 0.02), n={n}"] = false_ties / reps
    assert false_ties / reps <= 0.07


def test_four_equal_candidates_keep_reasonable_power_and_one_worse_one_is_not_tied():
    rng = np.random.default_rng(16)
    reps = 100
    # Informational (not asserted): at margin 0.03 the best-of-4 selection gap plus Holm over 3
    # comparisons leaves almost no power even for a fixed-best bootstrap (about 0.11); see the report.
    tight = 0
    for _ in range(reps):
        y, preds = simulate([0.90] * 4, 800, rng)
        tight += all(equivalent_set(boot_of(y, preds, rng), 0.03).values())
    RATES["k=4 equal: all tied, n=800, margin 0.03 (informational)"] = tight / reps
    all_tied = 0
    for _ in range(reps):
        y, preds = simulate([0.90] * 4, 800, rng)
        all_tied += all(equivalent_set(boot_of(y, preds, rng), 0.05).values())
    RATES["k=4 equal: all tied (power), n=800, margin 0.05"] = all_tied / reps
    assert all_tied / reps >= 0.3  # Holm over 3 comparisons costs power; informational floor
    false_ties = 0
    for _ in range(reps):
        y, preds = simulate([0.92, 0.92, 0.92, 0.72], 300, rng)  # m3 is 0.20 worse
        boot = boot_of(y, preds, rng)
        eq = equivalent_set(boot, 0.03)
        false_ties += eq["m3"] if boot.best != "m3" else 0
    RATES["k=4: clearly worse candidate tied, n=300"] = false_ties / reps
    assert false_ties / reps <= 0.05


def test_recommend_never_crowns_a_truly_worse_cheap_model_and_usually_picks_an_identical_one():
    rng = np.random.default_rng(17)
    reps, margin = 100, 0.03
    cost = {"m0": 10.0, "m1": 1.0}  # m1 is the cheap candidate
    wrong = 0
    for _ in range(reps):
        y, preds = simulate([0.90, 0.75], 300, rng)  # cheap model truly 0.15 worse
        boot = boot_of(y, preds, rng)
        rec = recommend(boot, equivalent_set(boot, margin), cost, margin)
        wrong += (rec.winner == "m1") if boot.best == "m0" else 0
    RATES["recommend crowns cheap model 0.15 worse, n=300"] = wrong / reps
    assert wrong / reps <= 0.05
    picked = 0
    for _ in range(reps):
        y, preds = simulate([0.90, 0.90], 800, rng)  # cheap model truly identical
        boot = boot_of(y, preds, rng)
        rec = recommend(boot, equivalent_set(boot, margin), cost, margin)
        picked += rec.winner == "m1"
    RATES["recommend picks identical cheap model, n=800"] = picked / reps
    assert picked / reps >= 0.5


def test_zz_print_measured_rates():
    print("\nMeasured rates (seeded):")
    for k, v in RATES.items():
        print(f"  {v:6.3f}  {k}")
