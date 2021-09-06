#!/bin/bash
for J in `seq -1 0.1 1`
do
STR="
#!/bin/bash\n\
#SBATCH -N 1\n\
#SBATCH -t 16:00:00\n\
#SBATCH --output=${J}.out\n\
#SBATCH --error=${J}.err\n\
\n\
python3 main.py --path_results "." --qlr 0.01 --acceptance_percentage 0.01 --n_qubits 8 --reps 1000 --qepochs 1000 --problem_config '{"problem":"XXZ","g":"0.75","J":"$J"}'
echo -e ${STR} | sbatch
done
