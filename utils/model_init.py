# get architecture
def get_arch(opt):
    arch = opt.arch
    class_num = opt.class_num

    print('You choose ' + arch + '...')
    if arch == 'ResDenseNet':
        from models import ResDenseNet
        model = ResDenseNet(nOUT=class_num)
    elif arch == 'ResUDenseNet':
        from models import ResUDenseNet
        model = ResUDenseNet(nOUT=class_num)

    return model


def get_arch_test(opt):
    class_num = opt.class_num

    from models import ResDenseNet
    from models import ResUDenseNet

    model_adult = ResDenseNet(nOUT=class_num)
    model_child = ResUDenseNet(nOUT=class_num)

    return model_adult, model_child


# get trainer and validator (train method)
def get_train_mode(opt):
    loss_type = opt.loss_type

    print('You choose ' + loss_type + 'trainer ...')
    if loss_type == 'base':  # multiple(joint) loss function
        from .trainer import base_train
        from .trainer import base_valid
        trainer = base_train
        validator = base_valid
    else:
        raise Exception("Loss type error!")

    return trainer, validator


# get loss function
def get_loss(opt):
    loss_oper = opt.loss_oper
    DEVICE = opt.device

    print('You choose ' + loss_oper + ' loss function ...')
    if loss_oper == 'base':
        import torch.nn as nn
        mae_loss = nn.L1Loss().to(DEVICE)

        return mae_loss
    else:
        raise Exception("Loss type error!")
