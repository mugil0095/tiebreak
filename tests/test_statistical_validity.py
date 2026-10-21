import numpy as np
import pytest
from scipy.stats import norm

from tiebreak.engine import ELIMINATION_FIRST_ROUND_MIN, ELIMINATION_MIN_N
from tiebreak.metrics import accuracy, macro_f1
from tiebreak.planner import clearly_worse
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


@pytest.mark.parametrize("n", [100, 300])
def test_false_tie_rate_for_macro_f1_at_the_margin_boundary_stays_small(n):
    """3 balanced classes; the competitor's errors are a superset of the best's (nested) and go to a uniformly
    random wrong class, so each class has precision = recall = accuracy and the TRUE macro-F1 of the best (0.90)
    exceeds the competitor's (0.88) by exactly the margin 0.02. The decision uses the real macro_f1 function."""
    rng = np.random.default_rng(300 + n)
    margin, reps = 0.02, 300
    false_ties = best_m0 = 0
    for _ in range(reps):
        y = np.arange(n) % 3
        u = rng.random(n)
        wrong = (y + rng.integers(1, 3, n)) % 3
        preds = {"m0": np.where(u < 0.10, wrong, y), "m1": np.where(u < 0.12, wrong, y)}
        boot = paired_bootstrap(y, preds, macro_f1, n_boot=400, seed=int(rng.integers(10**9)))
        if boot.best == "m0":
            best_m0 += 1
            false_ties += equivalent_set(boot, margin)["m1"]
    rate = cond_rate(false_ties, best_m0, reps)
    RATES[f"false tie | macro-F1, 3 classes, at boundary (loss = margin 0.02), n={n}"] = rate
    assert rate <= 0.07


@pytest.mark.parametrize("n", [100, 300])
def test_false_tie_rate_at_the_boundary_when_both_models_make_unique_errors(n):
    """Non-nested errors: P(only best wrong) = 0.04, P(only competitor wrong) = 0.06, P(both wrong) = 0.04,
    so accuracies are 0.92 vs 0.90 and b - c = margin * n in expectation with b > 0 and c > 0."""
    rng = np.random.default_rng(400 + n)
    margin, reps = 0.02, 300
    false_ties = best_m0 = 0
    for _ in range(reps):
        u = rng.random(n)
        y = np.ones(n, dtype=np.int64)
        m0_right = ~(u < 0.04) & ~((u >= 0.10) & (u < 0.14))  # wrong in [0, 0.04) (m0 only) and [0.10, 0.14) (both)
        m1_right = ~((u >= 0.04) & (u < 0.14))                # wrong in [0.04, 0.10) (m1 only) and [0.10, 0.14)
        preds = {"m0": m0_right.astype(np.int64), "m1": m1_right.astype(np.int64)}
        boot = boot_of(y, preds, rng, n_boot=400)
        if boot.best == "m0":
            best_m0 += 1
            false_ties += equivalent_set(boot, margin)["m1"]
    rate = cond_rate(false_ties, best_m0, reps)
    RATES[f"false tie | non-nested errors, at boundary (loss = margin 0.02), n={n}"] = rate
    assert rate <= 0.07


ELIMINATION_FALSE_DROP_BOUND = {ELIMINATION_MIN_N: 0.08, 200: 0.13}  # measured: see docs/safety-and-limits.md


@pytest.mark.parametrize("n", [ELIMINATION_MIN_N, 200])
def test_early_elimination_of_a_tied_candidate_at_the_shipped_settings(n):
    """5 truly tied candidates (accuracy 0.85); the engine's first round (max(50, n // 3) examples, only run at
    n >= 120) with clearly_worse's shipped defaults and the default margin 0.02. Counts runs in which ANY
    tied candidate would have been dropped."""
    rng = np.random.default_rng(500 + n)
    margin, reps = 0.02, 300
    first = max(ELIMINATION_FIRST_ROUND_MIN, n // 3)
    dropped_runs = 0
    for _ in range(reps):
        y, preds = simulate([0.85] * 5, n, rng)
        sub = rng.permutation(n)[:first]
        dropped = clearly_worse(y[sub], {k: v[sub] for k, v in preds.items()}, accuracy, margin,
                                eligible=set(preds), seed=int(rng.integers(10**9)))
        dropped_runs += bool(dropped)
    RATES[f"early elimination drops a tied candidate | 5 tied, n={n}, first round {first}"] = dropped_runs / reps
    # characterised, not guaranteed: strict thresholds still drop a tied model now and then with 5 candidates
    assert dropped_runs / reps <= ELIMINATION_FALSE_DROP_BOUND[n]


JUDGE_MARGIN = 0.25
JUDGE_SHIFT = 0.259  # latent shift for which the rounded, clipped 1-5 scores have a population mean gap of 0.25


def _judge_scores(n: int, rng):
    """Two judge-scored candidates with correlated, integer-valued scores (1 to 5); the first is better by
    JUDGE_MARGIN points on average, so the second sits exactly on the tie boundary."""
    q = rng.normal(0, 0.9, n)
    a = np.clip(np.round(3.3 + q + rng.normal(0, 0.8, n) * 0.5), 1, 5)
    b = np.clip(np.round(3.3 - JUDGE_SHIFT + q + rng.normal(0, 0.8, n) * 0.5), 1, 5)
    return a, b


def test_the_judge_score_gap_constant_matches_the_margin():
    a, b = _judge_scores(400_000, np.random.default_rng(1))
    assert abs((a - b).mean() - JUDGE_MARGIN) < 0.01


@pytest.mark.parametrize("n", [20, 100])
def test_false_tie_rate_for_judge_scores_at_the_margin_boundary_stays_small(n):
    """Judge scores are compared through the regression path (neg_mae against the top of the scale) with no
    discordance guard. At a true gap equal to the margin the bootstrap alone should falsely tie the worse
    candidate at about the nominal 5%; docs/how-it-works.md quotes the measured rates."""
    from tiebreak.metrics import neg_mae

    rng = np.random.default_rng(900 + n)
    reps, best_a, false_ties = 600, 0, 0
    y = np.full(n, 5.0)
    for _ in range(reps):
        a, b = _judge_scores(n, rng)
        boot = paired_bootstrap(y, {"A": a, "B": b}, neg_mae, n_boot=N_BOOT, seed=int(rng.integers(10**9)))
        if boot.best == "A":
            best_a += 1
            false_ties += equivalent_set(boot, JUDGE_MARGIN, guard=False)["B"]
    rate = cond_rate(false_ties, best_a, reps)
    RATES[f"false tie | judge scores (1-5), at boundary (gap = margin 0.25), bootstrap only, n={n}"] = rate
    assert rate <= 0.09
