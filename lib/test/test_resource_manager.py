
from typing import List, Tuple, Dict, Union, Optional

import os
import os.path as osp

import numpy as np
import trimesh

import torch

from lib.utils.net_utils import get_encode_net, get_render_net

class Test_Resource_Manager:
    tag_refine_net = "penone.dirs.diravg.seq+9+1+unet+5+2+16"
    tag_encode_net = "resunet3.16"

    def __init__(
            self,
            path_ckpt_encode:str,
            path_ckpt_render:str,
            device:torch.device
    ) -> None:
        assert osp.isfile(path_ckpt_encode), path_ckpt_encode
        assert osp.isfile(path_ckpt_render), path_ckpt_render
        self.device = device
        self.encode_net = Test_Resource_Manager.load_encoder_net_with_ckpt(path_ckpt_encode, device)
        self.render_net = Test_Resource_Manager.load_render_net_with_ckpt(path_ckpt_render, device)

    def get_encoder_net(self) -> (torch.nn.Module):
        return self.encode_net
    
    def get_render_net(self) -> (torch.nn.Module):
        return self.render_net
    
    def get_device(self) -> (torch.device):
        return self.device

    @staticmethod
    def check_ckpt_and_load(path_ckpt:str, net:torch.nn.Module, device:torch.device):
        assert osp.isfile(path_ckpt), path_ckpt
        print(f"Load ckpt from {path_ckpt}")
        pre_render_state_dict = torch.load(path_ckpt, map_location=device)

        state_dict_render = {}
        for key in net.state_dict().keys():
            if key in pre_render_state_dict.keys():
                if net.state_dict()[key].shape == pre_render_state_dict[key].shape:
                    state_dict_render[key] = pre_render_state_dict[key]
                else:
                    print(f"Not load {key} | Shape mismatch! {path_ckpt}")
            else:
                print(f"Not load {key} | Not in pre model! {path_ckpt}")

        net.load_state_dict(state_dict_render, strict=False)
        return

    @staticmethod
    def load_render_net_with_ckpt(path_ckpt:str, device:torch.device) -> (torch.nn.Module):
        
        render_net = get_render_net(
            Test_Resource_Manager.tag_refine_net, 
            Test_Resource_Manager.tag_encode_net
        )

        Test_Resource_Manager.check_ckpt_and_load(path_ckpt, render_net, device)
        render_net = render_net.to(device)

        return render_net
    
    @staticmethod
    def load_encoder_net_with_ckpt(path_ckpt:str, device:torch.device) -> (torch.nn.Module):
        encode_net = get_encode_net(Test_Resource_Manager.tag_encode_net)

        Test_Resource_Manager.check_ckpt_and_load(path_ckpt, encode_net, device)
        encode_net = encode_net.to(device)
        encode_net.eval()

        return encode_net

