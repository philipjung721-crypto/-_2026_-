import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# ===== 이 부분만 수정하세요 =====
N = 501                  # 전체 인원
Q = 3                    # 선택지 수
D = 5                   # 누적 점수가 -D 미만이면 전략 교체
STEPS = 30000            # 전체 라운드
BURN_IN = 0          # 통계에서 제외할 초기 라운드
RUNS = 10                 # 독립 반복 횟수
V = 1e-4                 # 복제 후 작은 변이의 분산
SEED = 20261108           # 재현용 난수 시드

# paper: 승리 +1, 패배 -1/(Q-1)
# balanced: 승리 +(Q-1), 패배 -1
# pm1: 승리 +1, 패배 -1
#논문에서 사용된 방법은 paper, 즉 우리도 paper를 사용할 것
PAYOFF = "paper"

MODES = ("random", "best")
# random: 임의의 다른 사람 모방
# best: 최고 누적 득점자 모방
# 필요하면 "self", "reset"도 추가 가능
# self: 자기 전략을 조금 수정
# reset: 완전히 새로운 무작위 전략

SAMPLE_EVERY = 50         # P(O) 계산용 표본 간격
BLOCK = 1          # 시간별 성능 그래프의 구간 길이

# balanced는 paper의 Q-1배 보상입니다.
# 같은 교체 조건을 유지하려면 D도 Q-1배로 바꾸세요.
# ===============================

LABELS = {
    "random": "Random imitation",
    "best": "Best-score imitation",
    "self": "Self mutation",
    "reset": "Random new strategy",
}
ENTROPY_BINS = 20  # 각 선택 확률의 구간 수: 20이면 구간 폭 0.05
import numpy as np
import pandas as pd

# ===== 실험 설정 =====
N = 501
Q = 3
D = 5
STEPS = 30000
BURN_IN = 0
RUNS = 10
V = 1e-4
SEED = 20261108

PAYOFF = "paper"
MODES = ("random", "best")

SAMPLE_EVERY = 50
ENTROPY_BINS = 20

LABELS = {
    "random": "Random imitation",
    "best": "Best-score imitation",
    "self": "Self mutation",
    "reset": "Random new strategy",
}


def donors(scores, ids, mode, rng):
    """자신을 제외한 모방 대상 선택."""
    if mode == "random":
        j = rng.integers(0, N - 1, len(ids))
        return j + (j >= ids)

    top = np.flatnonzero(scores == scores.max())

    if len(top) == 1:
        j = np.full(len(ids), top[0], dtype=int)
        mask = ids == top[0]

        if mask.any():
            other = np.flatnonzero(np.arange(N) != top[0])
            second = other[scores[other] == scores[other].max()]
            j[mask] = rng.choice(second, size=int(mask.sum()))

        return j

    loc = np.searchsorted(top, ids)
    inside = np.isin(ids, top)
    rank = (
        rng.random(len(ids)) * (len(top) - inside)
    ).astype(int)
    rank += inside & (rank >= loc)

    return top[rank]


def mutate(p, rng):
    """두 좌표 사이에서 확률을 이동시켜 합 1과 비음수성 보존."""
    p = p.copy()
    m = len(p)

    if m == 0 or V == 0:
        return p

    a = rng.integers(Q, size=m)
    b = rng.integers(Q - 1, size=m)
    b += b >= a

    row = np.arange(m)
    total = p[row, a] + p[row, b]
    proposal = p[row, a] + rng.normal(0, np.sqrt(V), m)

    positive = total > 0
    x = np.zeros(m)

    folded = np.remainder(
        proposal[positive], 2 * total[positive]
    )
    x[positive] = np.minimum(
        folded, 2 * total[positive] - folded
    )

    p[row, a], p[row, b] = x, total - x
    return p


def strategy_entropy(p, bins):
    """이산화한 전략벡터의 집단 내 빈도로 Shannon 엔트로피 계산."""
    cells = np.floor(
        np.clip(p, 0.0, 1.0) * bins
    ).astype(np.int64)
    cells = np.minimum(cells, bins - 1)

    _, counts = np.unique(
        cells, axis=0, return_counts=True
    )
    frequencies = counts / len(p)

    return float(
        -np.sum(frequencies * np.log(frequencies))
    )


def simulate(mode, run_id):
    def generator(k):
        return np.random.default_rng(
            np.random.SeedSequence([SEED, run_id, k])
        )

    initial = generator(0).dirichlet(np.ones(Q), N)
    p = initial.copy()

    action_rng, tie_rng, donor_rng, mutation_rng = [
        generator(k) for k in range(1, 5)
    ]

    # 분수 보상을 정수 단위로 계산
    scale = Q - 1 if PAYOFF == "paper" else 1
    win = Q - 1 if PAYOFF in ("paper", "balanced") else 1

    scores = np.zeros(N, dtype=np.int64)
    earnings = np.zeros(N, dtype=np.int64)
    measured_earnings = np.zeros(N, dtype=np.int64)

    counts = np.empty((STEPS, Q), dtype=int)
    group_units = np.empty(STEPS, dtype=np.int64)
    mean_o = np.empty(STEPS)
    entropy = np.empty(STEPS)
    replaced = np.zeros(STEPS, dtype=int)

    overlap = np.square(p).sum(1)
    samples = []

    for t in range(STEPS):
        # 해당 라운드의 선택 직전 전략 상태
        mean_o[t] = overlap.mean()
        entropy[t] = strategy_entropy(p, ENTROPY_BINS)

        if (
            t >= BURN_IN
            and (t - BURN_IN) % SAMPLE_EVERY == 0
        ):
            samples.append(overlap.copy())

        # 각 개인의 확률벡터에 따라 선택
        cdf = p.cumsum(1)
        cdf[:, -1] = 1.0
        actions = (
            action_rng.random(N)[:, None] >= cdf
        ).sum(1)

        count = np.bincount(actions, minlength=Q)

        # 최소 인원 선택지가 여러 개면 하나를 무작위로 선정
        tied = np.flatnonzero(count == count.min())
        winner = tied[
            min(
                int(tie_rng.random() * len(tied)),
                len(tied) - 1,
            )
        ]

        rewards = np.where(
            actions == winner, win, -1
        ).astype(np.int64)

        counts[t] = count
        group_units[t] = rewards.sum()

        earnings += rewards
        if t >= BURN_IN:
            measured_earnings += rewards

        scores += rewards

        # 점수가 역치 미만인 사람의 전략 교체
        ids = np.flatnonzero(scores < -D * scale)
        replaced[t] = len(ids)

        if len(ids):
            if mode == "reset":
                new = mutation_rng.dirichlet(
                    np.ones(Q), len(ids)
                )
            else:
                j = (
                    ids
                    if mode == "self"
                    else donors(scores, ids, mode, donor_rng)
                )
                new = mutate(p[j], mutation_rng)

            p[ids] = new
            overlap[ids] = np.square(new).sum(1)
            scores[ids] = 0

        if t % 1000 == 0:
            assert p.min() >= -1e-12
            assert np.allclose(p.sum(1), 1)

    sigma_q = np.square(counts - N / Q).mean(1)
    sigma_min = np.square(counts.min(1) - N / Q)
    payoff = group_units / scale
    sl = slice(BURN_IN, None)

    metrics = {
        "sigma_Q2": sigma_q[sl].mean(),
        "sigma_min2": sigma_min[sl].mean(),
        "mean_O": mean_o[sl].mean(),
        "strategy_entropy": entropy[sl].mean(),
        "total_payoff": group_units[sl].sum() / scale,
        "payoff_per_agent_per_round": payoff[sl].mean() / N,
    }

    return {
        "mode": mode,
        "run": run_id + 1,
        "initial": initial,
        "final": p,
        "counts": counts,
        "sigma_Q2": sigma_q,
        "sigma_min2": sigma_min,
        "mean_O": mean_o,
        "strategy_entropy": entropy,
        "O_samples": np.asarray(samples),
        "payoff": payoff,
        "earnings": earnings / scale,
        "measured_earnings": measured_earnings / scale,
        "replacements": replaced,
        "metrics": metrics,
        "cumulative_payoff": np.cumsum(group_units[sl]) / scale,
    }


def mean_se(values):
    """독립 실행 간 평균과 표준오차."""
    a = np.asarray(values)

    if len(a) > 1:
        se = a.std(axis=0, ddof=1) / np.sqrt(len(a))
    else:
        se = np.full(a.shape[1:], np.nan)

    return a.mean(axis=0), se


def run_comparison():
    integer_settings = {
        "N": N, "Q": Q, "STEPS": STEPS,
        "BURN_IN": BURN_IN, "RUNS": RUNS,
        "SAMPLE_EVERY": SAMPLE_EVERY,
        "ENTROPY_BINS": ENTROPY_BINS, "SEED": SEED,
    }
    for name, value in integer_settings.items():
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name}은 정수여야 합니다.")

    if N < 2 or Q < 2 or not (0 <= BURN_IN < STEPS):
        raise ValueError("N,Q >= 2, 0 <= BURN_IN < STEPS 필요")

    if min(RUNS, SAMPLE_EVERY, ENTROPY_BINS) < 1:
        raise ValueError("반복 횟수, 표본 간격, 구간 수는 양수여야 합니다.")

    if (
        not np.isfinite(D) or not np.isfinite(V)
        or D <= 0 or V < 0 or SEED < 0
    ):
        raise ValueError("유한한 D > 0, V >= 0, SEED >= 0 필요")

    if PAYOFF not in ("paper", "balanced", "pm1"):
        raise ValueError("PAYOFF 설정을 확인하세요.")

    if (
        not MODES
        or len(set(MODES)) != len(MODES)
        or any(m not in LABELS for m in MODES)
    ):
        raise ValueError("MODES 설정을 확인하세요.")

    rows, results = [], []

    for run_id in range(RUNS):
        for mode in MODES:
            r = simulate(mode, run_id)
            results.append(r)

            rows.append({
                "mode": mode,
                "run": run_id + 1,
                **r["metrics"],
            })

            print(
                f"{run_id + 1}/{RUNS} {mode}: "
                f"sigma_Q2={r['metrics']['sigma_Q2']:.4f}, "
                f"sigma_min2={r['metrics']['sigma_min2']:.4f}, "
                f"H={r['metrics']['strategy_entropy']:.4f}",
                flush=True,
            )

    per_run = pd.DataFrame(rows)
    summary_rows = []

    for mode in MODES:
        for metric in results[0]["metrics"]:
            values = per_run.loc[
                per_run["mode"] == mode, metric
            ].to_numpy()

            mean, se = mean_se(values)
            summary_rows.append({
                "mode": mode,
                "metric": metric,
                "mean": float(mean),
                "SE": float(se),
            })

    summary = pd.DataFrame(summary_rows)
    paired_rows = []

    if "random" in MODES and "best" in MODES:
        for metric in ("sigma_Q2", "sigma_min2"):
            a = per_run[
                per_run["mode"] == "random"
            ].set_index("run")[metric]
            b = per_run[
                per_run["mode"] == "best"
            ].set_index("run")[metric]

            mean, se = mean_se((b - a).to_numpy())
            paired_rows.append({
                "metric": metric,
                "best_minus_random": float(mean),
                "SE": float(se),
            })

    return results, per_run, summary, pd.DataFrame(paired_rows)


# ===== シミュレーション 실행 =====
results, per_run, summary, paired = run_comparison()

# 실행 당시 설정을 별도로 보관
simulation_config = {
    "N": N,
    "Q": Q,
    "D": D,
    "STEPS": STEPS,
    "BURN_IN": BURN_IN,
    "RUNS": RUNS,
    "V": V,
    "SEED": SEED,
    "PAYOFF": PAYOFF,
    "MODES": tuple(MODES),
    "SAMPLE_EVERY": SAMPLE_EVERY,
    "ENTROPY_BINS": ENTROPY_BINS,
    "LABELS": LABELS.copy(),
}

print("\n계산 완료. 이제 두 번째 셀을 실행하세요.")
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# ===== 그래프 설정: 여기만 바꾸고 이 셀을 다시 실행 =====
BLOCK = 1       # 성능 그래프의 평균 구간. 1이면 매 라운드 표시
HIST_BINS = 60     # 전문화도 히스토그램의 구간 수
SHOW_SE = True    # 표준오차 영역 표시 여부

simulation_config = {
    "N": N,
    "Q": Q,
    "D": D,
    "STEPS": STEPS,
    "BURN_IN": BURN_IN,
    "RUNS": RUNS,
    "V": V,
    "SEED": SEED,
    "PAYOFF": PAYOFF,
    "MODES": tuple(MODES),
    "SAMPLE_EVERY": SAMPLE_EVERY,
    "ENTROPY_BINS": ENTROPY_BINS,
    "LABELS": LABELS.copy(),
}

def visualize(results, summary, paired, config):
    # 시뮬레이션 실행 당시 설정 사용
    N = config["N"]
    Q = config["Q"]
    STEPS = config["STEPS"]
    BURN_IN = config["BURN_IN"]
    ENTROPY_BINS = config["ENTROPY_BINS"]
    MODES = config["MODES"]
    LABELS = config["LABELS"]

    if (
        not isinstance(BLOCK, (int, np.integer))
        or BLOCK < 1
        or not isinstance(HIST_BINS, (int, np.integer))
        or HIST_BINS < 1
    ):
        raise ValueError("BLOCK과 HIST_BINS는 양의 정수여야 합니다.")

    primary = ["sigma_Q2", "sigma_min2"]

    # 실행별 결과를 모방 방식별로 묶기
    by_mode = {
        mode: [r for r in results if r["mode"] == mode]
        for mode in MODES
    }

    # 그래프에서 실행 간 평균과 SE 계산
    def average_se(data):
        a = np.asarray(data)
        mean = a.mean(axis=0)
        se = (
            a.std(axis=0, ddof=1) / np.sqrt(len(a))
            if len(a) > 1
            else np.zeros_like(mean)
        )
        return mean, se

    def plot_average(ax, x, data, label):
        mean, se = average_se(data)
        line, = ax.plot(x, mean, label=label)

        if SHOW_SE and len(data) > 1:
            ax.fill_between(
                x, mean - se, mean + se,
                color=line.get_color(), alpha=0.15,
            )

    # ===== 표 출력 =====
    print(
        f"요약 통계 구간: {BURN_IN + 1}~{STEPS}라운드\n"
        "성능 지표: 두 값 모두 작을수록 좋습니다.\n"
        "SE는 독립 실행 간 표준오차입니다."
    )
    display(
        summary[
            summary["metric"].isin(primary)
        ].reset_index(drop=True)
    )

    if not paired.empty:
        print("\n차이 = best - random")
        print("양수이면 random의 지표가 더 낮습니다.")
        display(paired)

    print("\n보조 지표 및 전략 엔트로피")
    display(
        summary[
            ~summary["metric"].isin(primary)
        ].reset_index(drop=True)
    )

    # ===== 1. 전체 성능 비교 =====
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4), layout="constrained"
    )

    for ax, metric in zip(axes, primary):
        table = (
            summary[summary["metric"] == metric]
            .set_index("mode")
            .loc[list(MODES)]
        )

        ax.bar(
            range(len(MODES)),
            table["mean"],
            yerr=table["SE"].fillna(0) if SHOW_SE else None,
            capsize=5,
        )
        ax.set(
            xticks=range(len(MODES)),
            xticklabels=MODES,
            ylabel=metric,
            title=metric + " (lower is better)",
        )

    plt.show()

    # ===== 2. 개인별 전략 전문화도 분포 =====
    fig, ax = plt.subplots(
        figsize=(9, 4), layout="constrained"
    )
    bins = np.linspace(1 / Q, 1, HIST_BINS + 1)

    initial_o = np.concatenate([
        np.square(r["initial"]).sum(1)
        for r in by_mode[MODES[0]]
    ])

    ax.hist(
        np.clip(initial_o, 1 / Q, 1),
        bins=bins,
        density=True,
        histtype="step",
        color="black",
        label="Initial",
    )

    for mode in MODES:
        values = np.concatenate([
            r["O_samples"].ravel()
            for r in by_mode[mode]
        ])

        ax.hist(
            np.clip(values, 1 / Q, 1),
            bins=bins,
            density=True,
            histtype="step",
            label=LABELS[mode],
        )

    ax.set(
        xlabel="O = sum(p_q squared)",
        ylabel="P(O)",
        title=f"Specialization samples: rounds {BURN_IN + 1}-{STEPS}",
    )
    ax.legend()
    plt.show()

    # ===== 3. 1라운드부터의 성능 변화 =====
    starts = np.arange(0, STEPS, BLOCK)
    x = [
        a + (min(BLOCK, STEPS - a) + 1) / 2
        for a in starts
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4), layout="constrained"
    )

    for ax, metric in zip(axes, primary):
        for mode in MODES:
            data = [
                [
                    r[metric][a:a + BLOCK].mean()
                    for a in starts
                ]
                for r in by_mode[mode]
            ]
            plot_average(ax, x, data, LABELS[mode])

        ax.set(
            xlabel="Round",
            ylabel=metric,
            title=f"Performance over time (block={BLOCK})",
        )
        ax.legend(fontsize=8)

    plt.show()

    # ===== 4. 1라운드부터의 실제 누적득점 =====
    fig, ax = plt.subplots(
        figsize=(9, 4), layout="constrained"
    )
    rounds = np.arange(1, STEPS + 1)

    for mode in MODES:
        # 저장된 라운드별 득점을 누적. 시뮬레이션 재실행 없음.
        data = [
            np.cumsum(r["payoff"]) / N
            for r in by_mode[mode]
        ]
        plot_average(ax, rounds, data, LABELS[mode])

    ax.set(
        xlabel="Round",
        ylabel="Cumulative group payoff / N",
        title="Actual cumulative payoff from round 1",
    )
    ax.legend()
    plt.show()

    # ===== 5. 각 라운드의 전략 엔트로피 =====
    fig, ax = plt.subplots(
        figsize=(10, 4), layout="constrained"
    )

    for mode in MODES:
        data = [
            r["strategy_entropy"]
            for r in by_mode[mode]
        ]
        plot_average(ax, rounds, data, LABELS[mode])

    if BURN_IN > 0:
        ax.axvline(
            BURN_IN + 0.5,
            color="gray",
            linestyle="--",
            label="End of burn-in",
        )

    ax.set(
        xlabel="Round",
        ylabel="Strategy entropy H (nats)",
        title=f"Strategy diversity ({ENTROPY_BINS} bins per coordinate)",
    )
    ax.legend()
    plt.show()


# 저장된 결과로 출력만 수행
visualize(results, summary, paired, simulation_config)
