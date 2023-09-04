import torch

def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_model, criterion, device, config):
    model.train()
    list_losses = []
    list_losses_x = []
    list_losses_u = []
    # configs
    threshold = config.threshold

    train_iteration = len(unlabeled_trainloader) + 1  # all_test_img / batch_size + 1
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    for batch_idx in range(train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            _, (inputs_u_weak, inputs_u_strong) = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            _, (inputs_u_weak, inputs_u_strong) = next(unlabeled_train_iter)

        inputs = torch.cat((inputs_x, inputs_u_weak, inputs_u_strong), dim=0).to(device)
        targets_x = targets_x.to(device)

        # forward pass
        logits = model(inputs)
        logits_x = logits[:len(inputs_x)]
        logits_u_weak, logits_u_strong = logits[len(inputs_x):].chunk(2)
        del logits

        # Loss ground_truth
        Lx = criterion(logits_x, targets_x)
        # Pseudo label
        pseudo_label = torch.softmax(logits_u_weak.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(threshold).float()
        # Loss Aug
        Lu = (criterion(logits_u_strong, targets_u) * mask).mean()
        # Total Loss
        loss = Lx + 0.05 * Lu
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_model.update_params()
        # record loss
        list_losses.append(loss.item())
        list_losses_x.append(Lx.item())
        list_losses_u.append(Lu.item())

    losses = sum(list_losses) / len(list_losses)
    losses_x = sum(list_losses_x) / len(list_losses)
    losses_u = sum(list_losses_u) / len(list_losses)

    return losses, losses_x, losses_u

