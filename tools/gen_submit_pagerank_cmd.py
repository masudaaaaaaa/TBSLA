import os
import sys
import argparse
import math

Ns = 1
Ne = 4


#N = 32000
N = 800000
#N = 1500000
#N = 3000000

C = 300
machine = 'Poincare'
matrix_format = {'COO', 'CSR', 'ELL', 'SCOO'}
#matrix_format = {'COO', 'CSR', 'ELL', 'SCOO', 'DENSE'}
#matrix_format = {'COO', 'CSR', 'ELL', 'DENSE'}

def decomp(n):
  i = 2
  factors = []
  while i * i <= n:
    if n % i:
      i += 1
    else:
      n //= i
      factors.append(i)
  if n > 1:
    factors.append(n)
  return factors 

i = Ns
while i <= Ne:
  print('# nb nodes : ', i)
  factors = decomp(i * 16)
  for mf in matrix_format:
    g2 = 1
    for f in factors:
      g1 = int(i * 16 / g2)
      if g1 == 0: continue
      print(f'python tools/submit_pagerank.py --timeout 400 --matrix_dim {N} --format {mf} --nodes {i} --C {C} --machine {machine} --lang MPI --wall-time 1:00:00 --GR {g1} --GC {g2}')
      if g1 != g2:
        print(f'python tools/submit_pagerank.py --timeout 400 --matrix_dim {N} --format {mf} --nodes {i} --C {C} --machine {machine} --lang MPI --wall-time 1:00:00 --GR {g2} --GC {g1}')
      g2 *= f
  i *= 2
