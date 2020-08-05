# Prerequisites
```
pip install torchnet tensorboardX tensorboard
pip install opencv-python
```

# Usage
```
sh run_modis.sh
```

# tensorboard visualization
```
ssh -L1234:localhost:1234 GPU_machine_IP
tensorboard --port 1234 --logdir outputs_dir
```
