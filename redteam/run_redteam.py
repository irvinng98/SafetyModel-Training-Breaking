import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from agent import red_team_agent, RedTeamState

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.taxonomy import UNSAFE_CATEGORIES

os.makedirs("results", exist_ok=True)

all_results = []

for category in UNSAFE_CATEGORIES:
    print(f"\n=== Red-teaming category: {category} ===")

    initial_state = RedTeamState(
        target_category=category,
        attack_history=[],
        current_attack="",
        model_response="",
        attack_succeeded=False,
        iteration=0,
        strategy_notes="No attempts yet. Start broad and try different framings.",
    )

    final_state = red_team_agent.invoke(initial_state)

    for entry in final_state["attack_history"]:
        entry["category"] = category
        all_results.append(entry)

    category_results = [e for e in all_results if e["category"] == category]
    successes = sum(1 for e in category_results if e["succeeded"])
    print(f"  → {successes}/{len(category_results)} attacks succeeded for '{category}'")

df = pd.DataFrame(all_results)
df.to_csv("results/redteam_results.csv", index=False)

summary = df.groupby("category")["succeeded"].agg(
    successes="sum",
    total_attempts="count",
    success_rate="mean",
)
print("\n=== Red-Team Summary ===")
print(summary.to_string())
summary.to_csv("results/redteam_summary.csv")