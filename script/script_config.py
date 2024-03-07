
import os
import os.path as osp
import sys

path_folder_script = osp.dirname(__file__)
path_folder_hdhuman_repo = osp.abspath(osp.join(path_folder_script, osp.pardir))
path_folder_checkpoints = osp.abspath(osp.join(path_folder_hdhuman_repo, osp.pardir, 'checkpoints'))
path_folder_svs_repo = osp.abspath(osp.join(path_folder_hdhuman_repo, osp.pardir, 'StableViewSynthesis'))
path_folder_example_data = osp.abspath(osp.join(path_folder_hdhuman_repo, osp.pardir, 'example_data'))

assert osp.isdir(path_folder_hdhuman_repo), path_folder_hdhuman_repo
assert osp.basename(path_folder_hdhuman_repo) == 'HDhuman', path_folder_hdhuman_repo

sys.path.append(path_folder_hdhuman_repo)
