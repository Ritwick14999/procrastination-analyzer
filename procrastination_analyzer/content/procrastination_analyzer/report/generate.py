
import os

def generate_report(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    adv = data["advanced"]
    exp = data["explainability"]
    snippets = data["snippets"]
    meta = data.get("meta", {})

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Procrastination Pattern Analysis Report\n\n")

        if meta:
            f.write("## Metadata\n")
            for k, v in meta.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")

        f.write("## Summary\n")
        f.write(f"- Pattern: **{adv['type']}**\n")
        f.write(f"- Avoidance score: **{adv['perfectionism']}**\n")
        f.write(f"- Next-day risk: **{adv['risk']}**\n\n")

        f.write("## Signals\n")
        for k, v in exp.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n")

        f.write("## Suggestions\n\n")
        for s in snippets:
            f.write(f"### {s.get('title','Suggestion')}\n")
            f.write(f"- Category: {s.get('category')}\n")
            f.write(f"- Relevance: {s.get('score')}\n")
            if s.get("tags"):
                f.write(f"- Tags: {', '.join(s['tags'])}\n")
            f.write("\n")
            f.write(s.get("text", "") + "\n\n")
