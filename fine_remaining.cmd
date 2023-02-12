@REM # semi-sup
python fine.py --src_dataset=n-caltech101 --subset_len=10 experiments/SSL_Checkpoints/best_cnn_ncaltech101.ckpt
python fine.py --src_dataset=n-caltech101 --subset_len=25 experiments/SSL_Checkpoints/best_cnn_ncaltech101.ckpt




@REM #######################""

@REM # dvsgesture SNN-cnn
python fine.py --src_dataset=dvsgesture --subset_len=10 experiments/SSL_Checkpoints/best_snn_cnn_dvsgesture.ckpt
python fine.py --src_dataset=dvsgesture --subset_len=25 experiments/SSL_Checkpoints/best_snn_cnn_dvsgesture.ckpt

@REM # dvsgesture snn-CNN
python fine.py --src_dataset=dvsgesture --subset_len=10 --use_enc2 experiments/SSL_Checkpoints/best_cnn_snn_dvsgesture.ckpt
python fine.py --src_dataset=dvsgesture --subset_len=25 --use_enc2 experiments/SSL_Checkpoints/best_cnn_snn_dvsgesture.ckpt

@REM # ncaltech101 SNN-cnn
python fine.py --src_dataset=n-caltech101 --subset_len=10 experiments/SSL_Checkpoints/snn_cnn_ncaltech101.ckpt
python fine.py --src_dataset=n-caltech101 --subset_len=25 experiments/SSL_Checkpoints/snn_cnn_ncaltech101.ckpt

@REM # ncaltech101 snn-CNN
python fine.py --src_dataset=n-caltech101 --subset_len=10 --use_enc2 experiments/SSL_Checkpoints/cnn_snn_ncaltech101.ckpt
python fine.py --src_dataset=n-caltech101 --subset_len=25 --use_enc2 experiments/SSL_Checkpoints/cnn_snn_ncaltech101.ckpt

@REM #######################"


@REM # dvsgesture SNN-3dcnn
python fine.py --src_dataset=dvsgesture --subset_len=10 experiments/SSL_Checkpoints/best_3dcnn_snn_dvsgesture.ckpt
python fine.py --src_dataset=dvsgesture --subset_len=25 experiments/SSL_Checkpoints/best_3dcnn_snn_dvsgesture.ckpt

@REM # dvsgesture snn-3DCNN
python fine.py --src_dataset=dvsgesture --subset_len=10 --use_enc2 experiments/SSL_Checkpoints/best_3dcnn_snn_dvsgesture.ckpt
python fine.py --src_dataset=dvsgesture --subset_len=25 --use_enc2 experiments/SSL_Checkpoints/best_3dcnn_snn_dvsgesture.ckpt

@REM # ncaltech101 SNN-3dcnn
python fine.py --src_dataset=n-caltech101 --subset_len=10 experiments/SSL_Checkpoints/3dcnn_snn_ncaltech101.ckpt
python fine.py --src_dataset=n-caltech101 --subset_len=25 experiments/SSL_Checkpoints/3dcnn_snn_ncaltech101.ckpt

@REM # ncaltech101 snn-3DCNN
python fine.py --src_dataset=n-caltech101 --subset_len=10 --use_enc2 experiments/SSL_Checkpoints/3dcnn_snn_ncaltech101.ckpt
python fine.py --src_dataset=n-caltech101 --subset_len=25 --use_enc2 experiments/SSL_Checkpoints/3dcnn_snn_ncaltech101.ckpt


@REM #######################"

@REM # dvsgesture -> dailyAction SNN-cnn
python fine.py --src_dataset=dvsgesture --dest_dataset=daily_action_dvs experiments/SSL_Checkpoints/best_snn_cnn_dvsgesture.ckpt

@REM # dvsgesture -> dailyAction snn-CNN
python fine.py --src_dataset=dvsgesture --dest_dataset=daily_action_dvs --use_enc2 experiments/SSL_Checkpoints/best_cnn_snn_dvsgesture.ckpt

@REM # dvsgesture -> dailyAction SNN-3dcnn
python fine.py --src_dataset=dvsgesture --dest_dataset=daily_action_dvs experiments/SSL_Checkpoints/best_3dcnn_snn_dvsgesture.ckpt

@REM # dvsgesture -> dailyAction snn-3DCNN
python fine.py --src_dataset=dvsgesture --dest_dataset=daily_action_dvs --use_enc2 experiments/SSL_Checkpoints/best_3dcnn_snn_dvsgesture.ckpt