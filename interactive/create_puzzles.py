import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from sample import generate
from reward import compute_score, cpu_count
from add_student import add_student_measures
from datasets import Dataset

TEMPERATURE = 1.0
TOP_P = 0.98
MAX_TOKENS = 512

app = FastAPI()

@app.get("/create/{n}")
def create(n: int):
  prompts = [[{"role": "user", "content": "Generate a chess puzzle."}] for _ in range(n)]
  outputs = generate("teacher", prompts, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)
  xs = Dataset.from_list([{"solution_str": x[-1]['content'].split("</think>")[-1].strip()} for x in outputs])
  xs = xs.map(lambda x: compute_score(None, x['solution_str'], None), num_proc=cpu_count)
  xs = xs.filter(lambda x: x['valid'] and x['uniqueness'] > 0.5)
  xs = xs.map(add_student_measures)
  xs = xs.sort('top_move_rank', reverse=True)
  return xs.to_list()

app.mount("/assets", StaticFiles(directory="interactive/client/assets"), name="assets")
app.mount("/node_modules", StaticFiles(directory="interactive/client/node_modules"), name="node_modules")

@app.get("/")
async def index():
  return FileResponse("interactive/client/index.html")

@app.get("/browse")
async def browse_page():
  return FileResponse("interactive/client/index.html")

@app.get("/samples.json")
async def sample_json():
  return FileResponse("interactive/top100-samples.json")

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=1234)
