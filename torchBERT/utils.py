def print_loss_log(train_loss, val_loss, test_loss):
    with open('mlm_loss.txt', 'w') as f:
        for idx in range(len(train_loss)):
            f.write('epoch {:3d} | train loss {:5.2f}'.format(idx,
                                                              train_loss[idx]))
        for idx in range(len(val_loss)):
            f.write('epoch {:3d} | val loss {:5.2f}'.format(idx,
                                                            val_loss[idx]))
        f.write('test loss {:5.2f}'.format(test_loss))
