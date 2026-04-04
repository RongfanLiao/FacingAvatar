
/usr/local/miniconda3/envs/avatar/bin/python train_motion_transvae.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --epochs 5 \
  --lr 1e-5 \
  --batch_size 4 \
  --num_workers 4 \
  --video_canvas_size 400 \
  --log_interval 1 \
  --val_interval 1 \
  --checkpoint_dir checkpoints/motion_transvae_lookingface