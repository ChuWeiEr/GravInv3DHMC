#!/bin/bash
dir="picture"
if [ ! -d "$dir" ];then
mkdir $dir
echo "Successfully create folder $dir."
else
echo "Folder $dir already exist."
fi

name=1
for((i=0;i<=0;i=i+1));
do
  # input: number test & total number of chains
  python plot_model_global.py $i 2 > logoutfig_global_T$name.txt 2>&1 &
  name=$(($name+1))
done
