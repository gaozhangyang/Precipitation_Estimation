mkdir datasets
mkdir datasets/modis
cd datasets/modis
ln -s ../../dataloaders/datasets/dataset.npy dataset.npy
cd ../..
python main.py --name rain --model baseline --dataset modis --input_nc 6 --output_nc 7 --ignore_index -1 --gpu 0 --epochs 200


#python main.py --name rain --model selfconsist --dataset modis --input_nc 6 --output_nc 7 --ignore_index -1 --gpu 0 --x_drop 0.2 --epochs 400
