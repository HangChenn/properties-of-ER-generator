###===================
#!/bin/bash
#PBS -l select=1:ncpus=16:mem=128gb:pcmem=4gb -l walltime=18:00:00
#PBS -l cput=504:00:00
#PBS -q standard
#PBS -W group_list=kobourov
###-------------------

echo "Node name:"
hostname

cd /home/u29/hangchen/same_stat_100_test/4_tuple

module load python/3.5
python3 generate_same_stat_tuple.py
