import os
import json

root = "../results/test"
out = "./models.json"

models = []
for model_name in os.listdir(root):
    model_dir = os.path.join(root, model_name)
    if not os.path.isdir(model_dir):
        continue
    for sub in os.listdir(model_dir):
        full_path = os.path.join(model_dir, sub)
        if os.path.isdir(full_path):
            if model_name == "lp":
                models.append(f"{model_name}/{sub}")
            else:
                for sub_sub in os.listdir(full_path):
                    full_path_sub_sub = os.path.join(full_path, sub_sub)
                    if os.path.isdir(full_path_sub_sub):
                        models.append(f"{model_name}/{sub}/{sub_sub}")

with open(out, "w") as f:
    json.dump(models, f, indent=2)
