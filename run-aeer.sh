base_dir=$(pwd)
cd ./src/aeer/model/
nohup python3 latent_auto_encoder.py --epochs 100 --size 200 --corrupt 0.1 --novenue > ${base_dir}/result.txt &
