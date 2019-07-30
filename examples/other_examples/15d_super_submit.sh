#!/usr/bin/env bash

for i in {0..127}
do
  sbatch 15_d_submit.sh ${i}
done