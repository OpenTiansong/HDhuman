
# Net Config
src_img_skip = False
# point_avg_mode = "diravg"

train_device = "cuda:0"

learning_rate_egr = 1e-4
learning_rate_texture = 1e-3

num_workers = 4

background_black = True
background_white = False

coarse_depth = True

pre_model_path = None # f"./checkpoints/train_v6_coarse_depth_3/net_iter_{start_iteration}" # seldom use
pre_optim_path = None # f"./checkpoints/train_v6_coarse_depth_3/optim_iter_{start_iteration}" # seldom use

rank_mode = "pointdir"
pad_width = 32
batch_size = 1


## Twindom data config
Twindom_log_name = "twindom_depth_testing_2" #"Twindom_enc_gatimg_ren_0" #"Twindom_egr_1" #"ZZR_FtTex_1"#"ZZR_FtDR_L1loss_frame_858_0" # "test"
log_tensorboard_path = f"./log_tensorboard/{Twindom_log_name}/"
log_file_path = f"{log_tensorboard_path}/{Twindom_log_name}.log"

checkpoints_path = f"./checkpoints/{Twindom_log_name}"
img_path = f"./train_temp_results/{Twindom_log_name}"

Twindom_egr_start_iteration = 351000
Twindom_egr_train_iterations = 500000
Twindom_egr_freq_save_model = 1000
Twindom_egr_freq_save_img = 500

train_num_views = 6 # 10
ref_num_views = 4 # 4
load_size = 1024

random_scale_aug = True
random_flip_x = True

eval_ref_num_views = 4




