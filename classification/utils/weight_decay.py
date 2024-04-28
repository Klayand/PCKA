def add_weight_decay(model, lr, weight_decay=1e-5, skip_list=(), sort_params=False):
    decay = []
    no_decay = []
    named_params = list(model.named_parameters())
    if sort_params:
        named_params.sort(key=lambda x: x[0])
    for name, param in named_params:
        if not param.requires_grad:
            continue  # frozen weights
        skip = False
        for skip_name in skip_list:
            if skip_name.startswith('[g]'):
                if skip_name[3:] in name:
                    skip = True
            elif name == skip_name:
                skip = True
        if len(param.shape) == 1 or name.endswith(".bias") or skip:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0., 'initial_lr': lr},
        {'params': decay, 'weight_decay': weight_decay, 'initial_lr': lr}]