from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dqn import DQNAgent
from environment import ConnectFourEnv


def choose_agent_action(agent: DQNAgent, state: np.ndarray, current_player: int, valid_actions: list[int]) -> int:
    return int(agent.get_action(state * current_player, valid_actions, train=False))


def play_game(agent: DQNAgent | None, agent_player: int, rng: random.Random) -> dict:
    env = ConnectFourEnv(render_mode=None)
    state = env.reset()
    done = False
    steps = 0
    info = {}

    while not done:
        valid_actions = env.get_valid_actions()
        if agent is not None and env.current_player == agent_player:
            action = choose_agent_action(agent, state, env.current_player, valid_actions)
        else:
            action = rng.choice(valid_actions)
        state, reward, done, info = env.step(action)
        steps += 1

    winner = int(info.get("winner", 0))
    env.close()
    return {"winner": winner, "steps": steps}


def summarize_games(rows: list[dict], agent_player: int | None = None) -> dict:
    total = len(rows)
    if total == 0:
        return {}
    if agent_player is None:
        first_player_wins = sum(1 for row in rows if row["winner"] == 1)
        second_player_wins = sum(1 for row in rows if row["winner"] == -1)
        draws = sum(1 for row in rows if row["winner"] == 0)
        return {
            "games": total,
            "first_player_win_rate": round(first_player_wins / total, 4),
            "second_player_win_rate": round(second_player_wins / total, 4),
            "draw_rate": round(draws / total, 4),
            "avg_steps": round(sum(row["steps"] for row in rows) / total, 2),
        }
    wins = sum(1 for row in rows if row["winner"] == agent_player)
    losses = sum(1 for row in rows if row["winner"] == -agent_player)
    draws = sum(1 for row in rows if row["winner"] == 0)
    return {
        "games": total,
        "agent_player": agent_player,
        "agent_win_rate": round(wins / total, 4),
        "agent_loss_rate": round(losses / total, 4),
        "draw_rate": round(draws / total, 4),
        "avg_steps": round(sum(row["steps"] for row in rows) / total, 2),
    }


def write_markdown(report: dict, output: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Connect Four Evaluation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Seed: `{report['seed']}`",
        f"- Checkpoint evaluated: `{report['checkpoint']['path']}`",
        f"- Checkpoint tracked in git: `{report['checkpoint']['tracked_in_git']}`",
        "",
        "## Agent vs Random",
        "",
        f"- Overall win rate: `{summary['agent_overall_win_rate']}`",
        f"- Overall loss rate: `{summary['agent_overall_loss_rate']}`",
        f"- Overall draw rate: `{summary['agent_overall_draw_rate']}`",
        f"- Games: `{summary['agent_games']}`",
        "",
        "## Split By Side",
        "",
        f"- Agent as first player win rate: `{summary['agent_as_first']['agent_win_rate']}`",
        f"- Agent as second player win rate: `{summary['agent_as_second']['agent_win_rate']}`",
        "",
        "## Random Baseline",
        "",
        f"- First-player win rate: `{summary['random_vs_random']['first_player_win_rate']}`",
        f"- Second-player win rate: `{summary['random_vs_random']['second_player_win_rate']}`",
        f"- Draw rate: `{summary['random_vs_random']['draw_rate']}`",
        "",
        "## Claim Guidance",
        "",
        report["claim_guidance"],
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Connect Four checkpoint against random play.")
    parser.add_argument("--checkpoint", default="models/agent.pth")
    parser.add_argument("--games-per-side", type=int, default=100)
    parser.add_argument("--random-baseline-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=642)
    parser.add_argument("--output", default="artifacts/evaluation/connect_four_eval.json")
    parser.add_argument("--markdown-output", default="artifacts/evaluation/connect_four_eval.md")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    checkpoint = Path(args.checkpoint)
    agent = DQNAgent.load(str(checkpoint), train=False)

    first_rows = [play_game(agent, 1, rng) for _ in range(args.games_per_side)]
    second_rows = [play_game(agent, -1, rng) for _ in range(args.games_per_side)]
    baseline_rows = [play_game(None, 0, rng) for _ in range(args.random_baseline_games)]
    all_agent_rows = first_rows + second_rows
    wins = sum(1 for row in first_rows if row["winner"] == 1) + sum(
        1 for row in second_rows if row["winner"] == -1
    )
    losses = sum(1 for row in first_rows if row["winner"] == -1) + sum(
        1 for row in second_rows if row["winner"] == 1
    )
    draws = sum(1 for row in all_agent_rows if row["winner"] == 0)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "checkpoint": {
            "path": str(checkpoint),
            "exists": checkpoint.exists(),
            "tracked_in_git": False,
            "note": "The local checkpoint is ignored by git; reproduce by training or placing a checkpoint at this path.",
        },
        "summary": {
            "agent_games": len(all_agent_rows),
            "agent_overall_win_rate": round(wins / len(all_agent_rows), 4),
            "agent_overall_loss_rate": round(losses / len(all_agent_rows), 4),
            "agent_overall_draw_rate": round(draws / len(all_agent_rows), 4),
            "agent_as_first": summarize_games(first_rows, 1),
            "agent_as_second": summarize_games(second_rows, -1),
            "random_vs_random": summarize_games(baseline_rows),
        },
        "claim_guidance": (
            "This is a deterministic local checkpoint-vs-random evaluation. "
            "It supports claiming an evaluation harness and checked-in eval report, "
            "but the checkpoint itself is not tracked in git."
        ),
    }

    output = Path(args.output)
    markdown_output = Path(args.markdown_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, markdown_output)
    print(output)
    print(markdown_output)


if __name__ == "__main__":
    main()
