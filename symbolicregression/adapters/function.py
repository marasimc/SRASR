import torch

def disable_model_grad(model):
    r""" disable whole model's gradient """
    for name, param in model.named_parameters():
        param.requires_grad = False


def enable_module_grad(model, module_name):
    r""" enable a module's gridient caclulation by module name"""
    for name, param in model.named_parameters():
        if module_name in name:
            print('enable grad for : ', name)
            param.requires_grad = True