# Data
1. play with discretization (i.e., 3, 4 classes), and see if classes are balanced
2. double check mean, std
3. add more labeled points
4. add more images
5. double check if train/val split makes sense

# Visualization
0. how to visualize cloud channels?
1. how to visualize -1?
2. edit models/baseline.py:vis()
3. edit models/self_sup_consist.py:vis()

# Model
1. try U-Net

# Algorithm
1. play with lr and lr schedule
2. play with coefficients of aux_loss
3. try num_classes of binary classfications (BCELoss not CrossEntropyLoss)
