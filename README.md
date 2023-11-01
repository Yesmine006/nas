### Conda Environment Setting
```
conda create -n nas 
conda activate nas
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
```
### Spikingjelly Installation 
```

pip install spikingjelly
```
### Training

*  Run the following command
```
python search_snn.py  --exp_name 'cifar100_backward' --dataset 'cifar100'  --celltype 'backward' --batch_size 32 --num_search 1000 
