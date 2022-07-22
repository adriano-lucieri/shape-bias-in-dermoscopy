from torch import optim


def optimizer_selector(name, learning_rate, model, momentum=0.9, wd=0.0):

    if name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    elif name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=wd,
                                  initial_accumulator_value=0, eps=1e-10)
    elif name == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=wd)
    elif name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd,
                                amsgrad=False)
    elif name == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
    elif name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=wd,
                                  momentum=0, centered=False)
    elif name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    return optimizer


def lr_schedule_selector(optimizer, name='Plateau', plateau_factor=0.1, plateau_patience=10, step_size=10, step_gamma=0.1, plateau_threshold=1e-4):
    if name == 'Plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=plateau_factor, patience=plateau_patience,
                                                         threshold=plateau_threshold)
    elif name == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma, last_epoch=-1)
    elif name == 'None':
        scheduler = None

    return scheduler