import torch

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def save_checkpoint(epoch, model, optimizer, loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def save_checkpoint_stylegan2(g_module, d_module, g_ema, g_optim, d_optim, ada_aug_p, filename):
    torch.save(
        {
            "g": g_module.state_dict(),
            "d": d_module.state_dict(),
            "g_ema": g_ema.state_dict(),
            "g_optim": g_optim.state_dict(),
            "d_optim": d_optim.state_dict(),
            "ada_aug_p": ada_aug_p,
        },
        filename,
    )