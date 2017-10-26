base_dir=$(pwd)
cd ./src/aeer/model/
nohup python3 auto_encoder.py > ${base_dir}/result.txt &
