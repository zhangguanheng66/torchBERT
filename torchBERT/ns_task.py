import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from data import WikiText103
from model import NextSentenceTask
from utils import print_loss_log


def generate_next_sentence_data(whole_data):
    processed_data = []

    for item in whole_data:
        if len(item) > 1:
            # idx to split the text into two sentencd
            split_idx = torch.randint(1, len(item), size=(1, 1)).item()
            # Index 2 means same sentence label. Initial true int(1)
            processed_data.append([item[:split_idx], item[split_idx:], 1])

    # Random shuffle data to have args.frac_ns next sentence set up
    shuffle_idx1 = torch.randperm(len(processed_data))
    shuffle_idx2 = torch.randperm(len(processed_data))
    num_shuffle = int(len(processed_data) * args.frac_ns)
    shuffle_zip = list(zip(shuffle_idx1, shuffle_idx2))[:num_shuffle]
    for (i, j) in shuffle_zip:
        processed_data[i][1] = processed_data[j][0]
        processed_data[i][2] = int(0)  # Switch same sentence label to false 0
    return processed_data


def pad_next_sentence_data(batch):
    # Fix sequence length to args.bptt with padding or trim
    seq_list = []
    tok_type = []
    same_sentence_labels = []
    for item in batch:
        qa_item = torch.tensor(item[0] + [sep_id] + item[1] + [sep_id])
        if qa_item.size(0) > args.bptt:
            qa_item = qa_item[:args.bptt]
        elif qa_item.size(0) < args.bptt:
            qa_item = torch.cat((qa_item,
                                 torch.tensor([pad_id] * (args.bptt -
                                              qa_item.size(0)))))
        seq_list.append(qa_item)
        _tok_tp = torch.ones((qa_item.size(0)))
        _idx = min(len(item[0]) + 1, args.bptt)
        _tok_tp[:_idx] = 0.0
        tok_type.append(_tok_tp)
        same_sentence_labels.append(item[2])

    return torch.stack(seq_list).long().t().contiguous().to(device), \
        torch.stack(tok_type).long().t().contiguous().to(device), \
        torch.tensor(same_sentence_labels).long().contiguous().to(device)


###############################################################################
# Evaluating code
###############################################################################

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    batch_size = args.batch_size
    dataloader = DataLoader(data_source, batch_size=batch_size, shuffle=True,
                            collate_fn=pad_next_sentence_data)
    cls_id = data_source.vocab.stoi['<cls>']

    with torch.no_grad():
        for idx, (seq_input, tok_type, target_ns_labels) in enumerate(dataloader):
            # Add <'cls'> token id to the beginning of seq across batches
            seq_input = torch.cat((torch.tensor([[cls_id] * seq_input.size(1)]).long().to(device), seq_input))
            tok_type = torch.cat((torch.tensor([[0] * tok_type.size(1)]).long().to(device), tok_type))

            ns_labels = model(seq_input, token_type_input=tok_type)
            #print('ns_labels.size(), target_ns_labels.size()', ns_labels.size(), target_ns_labels.size())
            #print('ns_labels, target_ns_labels', ns_labels, target_ns_labels)
            loss = criterion(ns_labels, target_ns_labels)
            total_loss += loss.item()

    return total_loss / (len(data_source) // batch_size)


###############################################################################
# Training code
###############################################################################

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=pad_next_sentence_data)
    cls_id = train_dataset.vocab.stoi['<cls>']
#    softmax = torch.nn.Softmax(dim=-1) # print

    for idx, (seq_input, tok_type, target_ns_labels) in enumerate(dataloader):
        # Add <'cls'> token id to the beginning of seq across batches
        seq_input = torch.cat((torch.tensor([[cls_id] * seq_input.size(1)]).long().to(device), seq_input))
        tok_type = torch.cat((torch.tensor([[0] * tok_type.size(1)]).long().to(device), tok_type))
#        print('seq_input.size(), seq_input, tok_type.size(), tok_type', seq_input.size(), seq_input, tok_type.size(), tok_type)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        ns_labels = model(seq_input, token_type_input=tok_type)
#        print("batch, softmax(ns_labels).argmax(1), target_ns_labels", idx, softmax(ns_labels).argmax(1), target_ns_labels)
        loss = criterion(ns_labels, target_ns_labels)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        total_loss += loss.item()

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            train_loss_log.append(cur_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                  'ms/batch {:5.2f} | '
                  'loss {:8.5f} | ppl {:5.2f}'.format(epoch, idx,
                                                      len(train_dataset) // batch_size,
                                                      scheduler.get_last_lr()[0],
                                                      elapsed * 1000 / args.log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question-Answer fine-tuning task')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--lr', type=float, default=5,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=6, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='max. sequence length for context + question')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='ns_model.pt',
                        help='path to save the final model')
    parser.add_argument('--save-vocab', type=str, default='vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--bert-model', type=str,
                        help='path to save the pretrained bert')
    parser.add_argument('--frac_ns', type=float, default=0.5,
                        help='fraction of not next sentence')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###################################################################
    # Load data
    ###################################################################
    # Support WikiText103 for next sentence only
    try:
        vocab = torch.load(args.save_vocab)
    except:
        train_dataset, valid_dataset, test_dataset = WikiText103()
        old_vocab = train_dataset.vocab
        vocab = torchtext.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)
    pad_id = vocab.stoi['<pad>']
    sep_id = vocab.stoi['<sep>']

    train_dataset, valid_dataset, test_dataset = WikiText103(vocab=vocab,
                                                             single_line=False)
    train_dataset.data = generate_next_sentence_data(train_dataset.data)
    valid_dataset.data = generate_next_sentence_data(valid_dataset.data)
    test_dataset.data = generate_next_sentence_data(test_dataset.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###################################################################
    ###################################################################
    # Build the model
    ###################################################################

    pretrained_bert = torch.load(args.bert_model)
    model = NextSentenceTask(pretrained_bert).to(device)

    criterion = nn.CrossEntropyLoss()

    ###################################################################
    # Loop over epochs.
    ###################################################################

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    best_val_loss = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(valid_dataset)
        val_loss_log.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s '
              '| valid loss {:8.5f} | '.format(epoch,
                                               (time.time() - epoch_start_time),
                                               val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            scheduler.step()

    ###################################################################
    # Load the best saved model.
    ###################################################################
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    ###################################################################
    # Run on test data.
    ###################################################################
    test_loss = evaluate(test_dataset)
    print('=' * 89)
    print('| End of training | test loss {:8.5f}'.format(
        test_loss))
    print('=' * 89)
    print_loss_log(train_loss_log, val_loss_log, test_loss)

    with open(args.save, 'wb') as f:
        torch.save(model.bert_model, f)
#python qa_task.py --bert-model squad_vocab_pretrained_bert.pt --epochs 2 --save-vocab squad_vocab.pt
