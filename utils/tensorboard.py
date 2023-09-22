from tensorboardX import SummaryWriter


class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)

    def log_train_loss(self, loss_type, train_loss, step):
        self.add_scalar('train_{}_loss'.format(loss_type), train_loss, step)

    def log_valid_loss(self, loss_type, valid_loss, step):
        self.add_scalar('valid_{}_loss'.format(loss_type), valid_loss, step)
