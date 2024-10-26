#!/bin/bash
dir=(modeldata result)
for((j=0;j<=1;j++));
do
if [ ! -d "${dir[$j]}" ];then
mkdir ${dir[$j]}
echo "Successfully create folder ${dir[$j]}."
else
echo "Folder ${dir[$j]} already exist."
fi
done

name=0
for((i=0;i<=1;i++));
do
nohup mpiexec -n 2 python main_real.py $i > logout_T$name.txt 2>&1 &
name=$(($name+1))
done

