import logging


def adjustLR(optimizer, epoch, config):
    lr = config.train.lr
    for stage in config.train.stage_epochs:
        if epoch + 1 >= stage:
            lr /= config.train.lr_decay
    logging.info("adjust lr to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
