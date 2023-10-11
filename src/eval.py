import concurrent.futures as f
import ast

from datasets import DatasetDict, load_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate
import pandas as pd
from datasets import Dataset

import os

file_path = "./data/eval.xlsx"

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} has been deleted.")
else:
    print(f"The file {file_path} does not exist.")

to_eval = pd.read_excel("./data/results.xlsx")
to_eval['contexts'] = to_eval['contexts'].apply(ast.literal_eval)

d = Dataset.from_pandas(to_eval)

print(d.shape, d.column_names)


result = evaluate(
    d,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        #context_recall,
        #harmfulness,
    ],
)

df = result.to_pandas()
df.to_excel(file_path, index=False)
