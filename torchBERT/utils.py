def print_loss_log(file_name, train_loss, val_loss, test_loss):
    with open(file_name, 'w') as f:
        for idx in range(len(train_loss)):
            f.write('epoch {:3d} | train loss {:8.5f}'.format(idx + 1,
                                                              train_loss[idx]) + '\n')
        for idx in range(len(val_loss)):
            f.write('epoch {:3d} | val loss {:8.5f}'.format(idx + 1,
                                                            val_loss[idx]) + '\n')
        f.write('test loss {:8.5f}'.format(test_loss) + '\n')
