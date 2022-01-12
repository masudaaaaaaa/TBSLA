import json
import sys

def get_stats(json_input_file):
  stats = dict()
  with open(json_input_file) as fp:
    for cnt, line in enumerate(fp):
      line = line.strip()
      if not line.startswith("{"): continue
      mydict = json.loads(line)
      key = (mydict["lang"], mydict["format"], mydict["nodes"])
      if key not in stats:
        stats[key] = dict()
      if mydict["success"] not in stats[key]:
        stats[key][mydict["success"]] = 0
      stats[key][mydict["success"]] += 1
  return stats

if len(sys.argv) == 2:
  r = get_stats(sys.argv[1])
  print("(lang, format, nodes)")
  for k in sorted(list(r.keys())):
    print(k, "  ->  true : ", r[k].get("true", 0), "  false : ",r[k].get("false", 0))
else:
  print(sys.argv[0], "json_file_name")
