import json, collections
cap = {}
for l in open("jobs_v1/jobs_cap_v1.jsonl"):
    j = json.loads(l); cap[j["job"]] = j
kept = {json.loads(l)["job"] for l in open("gen_v1r2/kept.jsonl")}
oldk = {json.loads(l)["job"] for l in open("gen_v1r/kept.jsonl")}
for name in ("jobs_v1/jobs_g1.jsonl", "jobs_v1/jobs_w16k.jsonl"):
    c = collections.Counter(); t = collections.Counter()
    fresh = []
    for l in open(name):
        j = json.loads(l); c[j["source"]] += 1
        if j["job"] not in cap: t[j["source"], j["type"]] += 1; fresh.append(l)
    print(name, dict(c))
    print("  fresh", sorted(t.items()))
    if name.endswith("g1.jsonl"): keep = fresh
retry = [cap[k] for k in oldk - kept]
print("retry", len(retry), collections.Counter(j["source"] for j in retry))
with open("jobs_v1/jobs_v2.jsonl", "w") as f:
    for j in retry: f.write(json.dumps(j, ensure_ascii=False) + "\n")
    f.writelines(keep)
print("jobs_v2", len(retry) + len(keep))
