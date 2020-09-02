from Tools.generate_mp4 import Draw

if __name__ =='__main__':
    cmd={'iden_model_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/identification/005/epoch_28_step_28.pt',
        'esti_model_path':'',
        'train_path':'/usr/commondata/weather/dataset_release/IR_dataset_QingHua/',
        'step':3,
        'data_X':'X_val_hourly.npz',
        'data_Y':'Y_val_hourly.npz',
        'save_path':'/usr/commondata/weather/code/Precipitation_Estimation/results/Visualization/iden_global',
        'save_name':'val',
        'specify_task':'identification'
        }
    draw = Draw()
    draw.generate_final_surface_MP4(**cmd)