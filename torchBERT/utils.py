import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp


def setup(rank, world_size, seed):
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['WORLD_SIZE'] = str(2)
    # initialize the process group
    print("os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'], rank, os.environ['SLURM_PROCID']: ", os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'], rank, os.environ['SLURM_PROCID'])
    dist.init_process_group("nccl", init_method='env://', world_size=2)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(seed)


def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, args):
    mp.spawn(demo_fn,
             args=(args,),
             nprocs=args.world_size,
             join=True)


def print_loss_log(file_name, train_loss, val_loss, test_loss, args=None):
    with open(file_name, 'w') as f:
        if args:
            for item in args.__dict__:
                f.write(item + ':    ' + str(args.__dict__[item]) + '\n')
        for idx in range(len(train_loss)):
            f.write('epoch {:3d} | train loss {:8.5f}'.format(idx + 1,
                                                              train_loss[idx]) + '\n')
        for idx in range(len(val_loss)):
            f.write('epoch {:3d} | val loss {:8.5f}'.format(idx + 1,
                                                            val_loss[idx]) + '\n')
        f.write('test loss {:8.5f}'.format(test_loss) + '\n')
