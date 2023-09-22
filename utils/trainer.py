import torch
from .progress import Bar, np

def base_train(model, train_loader, loss_calculator, optimizer, writer,
               EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()

    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)

        if opt.mode == 'adult':
            outputs = 103 * torch.sigmoid(outputs) + 19
        elif opt.mode == 'child':
            outputs = 9 * torch.sigmoid(outputs)

        loss = loss_calculator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('Train Loss', train_loss, EPOCH)

    return train_loss


def base_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    batch_num = 0

    mae_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)

            if opt.mode == 'adult':
                outputs = 103 * torch.sigmoid(outputs) + 19
            elif opt.mode == 'child':
                outputs = 9 * torch.sigmoid(outputs)

            loss = loss_calculator(outputs, targets)

            valid_loss += loss
        valid_loss /= batch_num
        mae_loss /= batch_num
    # tensorboard
    writer.log_valid_loss('Validation Loss', valid_loss, EPOCH)

    return valid_loss

