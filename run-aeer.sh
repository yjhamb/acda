base_dir=$(pwd)
cd ./src
python3 ./acda/main.py --epochs 100 --size 200 --corrupt 0.1 --novenue > ${base_dir}/result.txt
