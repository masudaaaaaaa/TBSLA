import os
import sys
import argparse
import re
import collections
import operator

parser = argparse.ArgumentParser()
parser.add_argument("--machinefile", dest="machinefile", help="Machine file to reorder", type=str, required=True)
parser.add_argument("--output", "-o", dest="output", help="Machine file reordered", type=str, required=True)
args = parser.parse_args()

with open(args.machinefile, "r") as f:
  lines = f.readlines()
  for i in range(len(lines)):
    lines[i] = re.sub("\s", "", lines[i])
  counts = collections.Counter(lines)
  sorted_counts = sorted(counts.items(), key=operator.itemgetter(1))

  lines = []
  lines.append(sorted_counts[0][0])
  for i in range(1, len(sorted_counts)):
    for j in range(sorted_counts[i][1]):
      lines.append(sorted_counts[i][0])
  for j in range(1, sorted_counts[0][1]):
    lines.append(sorted_counts[0][0])
  for i in range(len(lines)):
    lines[i] = lines[i] + '\n'

  with open(args.output, "w") as fout:
    fout.writelines(lines)
