from datasets import Dataset

out = []
for _ in range(1024):
  out.append({
    "data_source": "puzzle",
    "prompt": [{"role": "user", "content": "Generate a chess puzzle."}],
    "reward_model": {"style": "rule", "ground_truth": None},
    "extra_info": {
      "need_tools_kwargs": False,
    }
  })

out = Dataset.from_list(out)
out.to_parquet("data/rl-default-1024-dec28.parquet")
