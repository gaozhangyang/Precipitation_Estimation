# Prerequisites
```
pip install torchnet tensorboardX tensorboard tensorflow
pip install opencv-python
```

# Usage
```
python main.py
```

# tensorboard visualization
```
ssh -L1234:localhost:1234 GPU_machine_IP
tensorboard --port 1234 --logdir outputs_dir
```
