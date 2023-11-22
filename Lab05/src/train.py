import Lab05.dsntnn as dsntnn

def train(model, optimizer, n_epochs, loss_fn, data_loader, validation_loader, device, plot_file, save_file, completed_epochs=0):
    print("Starting training...")
    model.to(device=device)
    train_losses = []
    validation_losses = []
    train_accuracy = []
    validation_accuracy = []
    for epoch in range(completed_epochs, n_epochs):
        print("Epoch", epoch)

        model.train()
        loss_train = 0.0
        loss_val = 0.0
        acc_train = 0.0
        acc_val = 0.0
        for imgs, target_coords in data_loader:
            imgs = imgs.to(device=device)
            coords = coords.to(device=device)

            # forward propagation
            coords, heatmaps = model(imgs)

            # calculate losses
            euclidean_loss = dsntnn.euclidean_losses(coords, target_coords)
            # the
            reg_loss = dsntnn.js_reg_losses(heatmaps, target_coords, sigma_t=1.0)

            # calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            acc_train += (predictions == labels).sum().item()/len(labels) * 100


            # reset optimizer gradients to zero
            optimizer.zero_grad()
            # calculate the loss gradients
            loss.backward()
            # iterate the optimization, based on the loss gradients
            optimizer.step()  # iterate the optimization, based on the loss gradients
            loss_train += loss.item()  # update the value of losses

        acc_train = acc_train/len(data_loader)
        train_accuracy.append(acc_train)

        # Calculate validation loss
        model.eval()
        with torch.no_grad():
            for imgs, labels in validation_loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                # forward propagation
                outputs = model(imgs)
                # calculate loss
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=2)
                loss = loss_fn(outputs, labels_one_hot.float())
                loss_val += loss.item()
                # calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                acc_val += (predictions == labels).sum().item() / len(labels) * 100

        acc_val = acc_val/len(validation_loader)
        validation_accuracy.append(acc_val)

        print('{} Epoch {}, Training loss {}, Validation Loss {}, Training Accuracy {}, Validation Accuracy {}'.format(datetime.datetime.now(),
                                                                                                                       epoch,
                                                                                                                       loss_train / len(data_loader),
                                                                                                                       loss_val / len(validation_loader),
                                                                                                                       acc_train,
                                                                                                                       acc_val))

        train_losses += [loss_train / len(data_loader)]  # update value of losses
        validation_losses += [loss_val / len(validation_loader)]

        # save the model and optimizer
        path = "./history/model_" + str(epoch) + ".pth"
        opt_path = "./history/optimizer_" + str(epoch) + ".pth"
        print("Saving model to: " + path)
        print("Saving optimizer to: " + opt_path)
        torch.save(model.state_dict(), path)
        torch.save(optimizer.state_dict(), opt_path)

        loss_file = open("./history/losses.txt", "a+")
        loss_file.write(str(train_losses[-1]) + " " + str(validation_losses[-1]) + "\n")
        loss_file.close()


    # plot the losses_train
    if save_file != None:
        torch.save(model.state_dict(), save_file)
    if plot_file != None:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(train_losses, label='train')
        plt.plot(validation_losses, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        plt.savefig(plot_file)