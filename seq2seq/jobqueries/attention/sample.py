import argparse
import copy
from main import *
import torch
import time
from functools import partial

from greedy import greedy
from beam import beam
from variable_beam import variable_beam
from variable_beam_stream import variable_beam_stream

DECODER_METHOD = [greedy, None, beam, variable_beam, variable_beam_stream, partial(variable_beam_stream, min_prog=True)]

def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def do_generate(encoder, decoder, attention_decoder, enc_w_list, word_manager, form_manager, opt, using_gpu, checkpoint):
    # initialize the rnn state to all zeros
    enc_w_list.append(word_manager.get_symbol_idx('<S>'))
    enc_w_list.insert(0, word_manager.get_symbol_idx('<E>'))
    end = len(enc_w_list)
    prev_c  = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    prev_h  = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    enc_outputs = torch.zeros((1, end, encoder.hidden_size), requires_grad=False)
    if using_gpu:
        prev_c = prev_c.cuda()
        prev_h = prev_h.cuda()
        enc_outputs = enc_outputs.cuda()
    # TODO check that c,h are zero on each iteration
    # reversed order
    for i in range(end-1, -1, -1):
        # TODO verify that this matches the copy_table etc in sample.lua
        cur_input = torch.tensor(np.array(enc_w_list[i]), dtype=torch.long)
        if using_gpu:
            cur_input = cur_input.cuda()
        prev_c, prev_h = encoder(cur_input, prev_c, prev_h)
        enc_outputs[:, i, :] = prev_h
    # decode
    if opt.sample == 0 or opt.sample == 1:
        text_gen = []
        if opt.gpuid >= 0:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long).cuda()
        else:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long)
        while True:
            prev_c, prev_h = decoder(prev_word, prev_c, prev_h)
            pred = attention_decoder(enc_outputs, prev_h)
            # log probabilities from the previous timestamp
            if opt.sample == 0:
                # use argmax
                _, _prev_word = pred.max(1)
                prev_word = _prev_word.resize(1)
            if (prev_word[0] == form_manager.get_symbol_idx('<E>')) or (len(text_gen) >= checkpoint["opt"].dec_seq_length):
                break
            else:
                text_gen.append(prev_word[0])
        return text_gen



if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-temperature', type=int, default=1, help='temperature of sampling')
    main_arg_parser.add_argument('-sample', type=int, default=0, help='0 for Greedy, 2 for Fixed, 3 for Var-Batch, 4 for Var-FIFO, 5 for Var-Stream')
    main_arg_parser.add_argument('-beam_size', type=int, default=10, help='beam size')
    main_arg_parser.add_argument('-ap', type=float, default=2.5, help='beam size')
    main_arg_parser.add_argument('-mc', type=int, default=3, help='beam size')
    main_arg_parser.add_argument('-display', type=int, default=1, help='whether display on console')
    main_arg_parser.add_argument('-data_dir', type=str, default='../data/', help='data path')
    main_arg_parser.add_argument('-input', type=str, default='test.t7', help='input data filename')
    main_arg_parser.add_argument('-output', type=str, default='output/seq2seq_attention_output.txt', help='input data filename')
    main_arg_parser.add_argument('-model', type=str, default='checkpoint_dir/model_seq2seq_attention', help='model checkpoint to use for sampling')
    main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')
    main_arg_parser.add_argument('-batch_size',type=int,default=10,help='decoding batch size')

    # parse input params
    args = main_arg_parser.parse_args()
    using_gpu = False
    if args.gpuid > -1:
        using_gpu = True
    # load the model checkpoint
    checkpoint = torch.load(args.model)
    encoder = checkpoint["encoder"]
    decoder = checkpoint["decoder"]
    attention_decoder = checkpoint["attention_decoder"]
    if using_gpu:
        encoder, decoder, attention_decoder = encoder.cuda(), decoder.cuda(), attention_decoder.cuda()
    # put in eval mode for dropout
    encoder.eval()
    decoder.eval()
    attention_decoder.eval()
    # initialize the vocabulary manager to display text
    managers = pkl.load( open("{}/map.pkl".format(args.data_dir), "rb" ) )
    word_manager, form_manager = managers
    # load data
    data = pkl.load(open("{}/test.pkl".format(args.data_dir), "rb"))
    data = sorted(data, key=lambda x: x[0])
    reference_list = []
    candidate_list = []
    with open(args.output, "w") as output:
        total_iters, total_expansions = 0, 0
        if args.sample == 0:
            args.batch_size *= args.beam_size
        step_size = len(data) if args.sample in [4, 5] else args.batch_size
        start_time = time.time()
        for i in range(0, len(data), step_size):
            x = data[i:i+step_size]
            references = [d[1] for d in x]
            candidates, iters, expansions = DECODER_METHOD[args.sample](encoder, decoder, attention_decoder, [d[0] for d in x], word_manager, form_manager, args, using_gpu, checkpoint)
            candidates = [[[int(c) for c in candidate] for candidate in clist] for clist in candidates]
            total_iters += iters
            total_expansions += expansions

            diffs = [[sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== "(") \
                    - sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== ")") for candidate in clist] for clist in candidates]
            for j in range(len(candidates)):
                for beam_index in range(len(candidates[j])):
                    if diffs[j][beam_index] > 0:
                        for _ in range(diffs[j][beam_index]):
                            candidates[j][beam_index].append(form_manager.symbol2idx[")"])
                    elif diffs[j][beam_index] < 0:
                        candidates[j][beam_index] = candidates[j][beam_index][:diffs[j][beam_index]]

            ref_strs = [convert_to_string(reference, form_manager) for reference in references]
            cand_strs = [[convert_to_string(candidate, form_manager) for candidate in clist] for clist in candidates]

            reference_list += references
            candidate_list += candidates
            # print to console
            if args.display > 0:
                print("results: ")
                print(ref_strs)
                print(cand_strs)
                print(' ')
            for ref_str, cand_str in zip(ref_strs, cand_strs):
                output.write("{}\n".format(ref_str))
                output.write("{}\n".format(cand_str))
        end_time = time.time()

        val_acc = util.compute_tree_accuracy([[c[0]] for c in candidate_list], reference_list, form_manager)
        oracle_val_acc = util.compute_tree_accuracy(candidate_list, reference_list, form_manager)
        print(("ACCURACY = {}\n".format(val_acc)))
        print(("ORACLE ACCURACY = {}\n".format(oracle_val_acc)))
        print(('iters: ' + str(total_iters)))
        print(('expansions: ' + str(total_expansions)))
        print(('seconds: ' + str(end_time - start_time)))
        output.write("ACCURACY = {}\n".format(val_acc))
        output.write("ORACLE ACCURACY = {}\n".format(oracle_val_acc))
        output.write('iters: ' + str(total_iters))
        output.write('expansions: ' + str(total_expansions))
        output.write(('seconds: ' + str(end_time - start_time)))