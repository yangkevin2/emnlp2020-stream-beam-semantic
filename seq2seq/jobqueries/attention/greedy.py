import torch
import numpy as np

def greedy(encoder, decoder, attention_decoder, enc_w_list, word_manager, form_manager, opt, using_gpu, checkpoint):
    iters = 0
    expansions = 0
    # initialize the rnn state to all zeros
    device = 'cuda:' + str(opt.gpuid) if using_gpu else 'cpu'
    batch_size = len(enc_w_list)
    for ewl in enc_w_list:
        ewl.append(word_manager.get_symbol_idx('<S>'))
        ewl.insert(0, word_manager.get_symbol_idx('<E>'))
    lengths = torch.tensor([len(ewl) for ewl in enc_w_list]).to(device)
    end = lengths.max()
    for i in range(len(enc_w_list)):
        if len(enc_w_list[i]) < end:
            enc_w_list[i] = [0 for _ in range(end - len(enc_w_list[i]))] + enc_w_list[i]
    prev_c  = torch.zeros((batch_size, encoder.hidden_size), requires_grad=False)
    prev_h  = torch.zeros((batch_size, encoder.hidden_size), requires_grad=False)
    enc_outputs = torch.zeros((batch_size, end, encoder.hidden_size), requires_grad=False) # batch x seq x hidden
    save_prev_c = torch.zeros((batch_size, end, encoder.hidden_size), requires_grad=False)
    if using_gpu:
        prev_c = prev_c.cuda()
        prev_h = prev_h.cuda()
        enc_outputs = enc_outputs.cuda()
    # TODO check that c,h are zero on each iteration
    # reversed order
    inputs = torch.tensor(enc_w_list, dtype=torch.long) # batch x seq
    if using_gpu:
        inputs = inputs.cuda()
    for i in range(end-1, -1, -1):
        # TODO verify that this matches the copy_table etc in sample.lua
        cur_input = inputs[:, i]
        prev_c, prev_h = encoder(cur_input, prev_c, prev_h)
        enc_outputs[:, i, :] = prev_h
        save_prev_c[:, i, :] = prev_c
    #encoder_outputs = torch.stack(encoder_outputs).view(-1, end, encoder.hidden_size)
    # decode
    prev_h = enc_outputs[torch.arange(batch_size), end-lengths, :] # batch x hidden
    prev_c = save_prev_c[torch.arange(batch_size), end-lengths, :] # batch x hidden
    if using_gpu:
        prev_h, prev_c = prev_h.cuda(), prev_c.cuda()
    if opt.sample == 0 or opt.sample == 1:
        end_indices = torch.tensor([1e8 for _ in range(batch_size)], dtype=torch.long)
        text_gen = []
        prev_word = torch.zeros(batch_size, dtype=torch.long) + form_manager.get_symbol_idx('<S>')
        if opt.gpuid >= 0:
            prev_word = prev_word.cuda()
        count = 0
        while True:
            iters += 1
            expansions += (end_indices == 1e8).long().sum() # only count valid beams that aren't completed
            prev_c, prev_h = decoder(prev_word, prev_c, prev_h)
            pred = attention_decoder(enc_outputs, prev_h, lengths)
            #print("prediction: {}\n".format(pred))
            # log probabilities from the previous timestamp
            if opt.sample == 0:
                # use argmax
                _, prev_word = pred.max(1) # batch
                # prev_word = _prev_word.resize(1)
            new_end = (prev_word == form_manager.get_symbol_idx('<E>')).nonzero().flatten()
            if count >= checkpoint["opt"].dec_seq_length: # break out
                new_end = list(range(batch_size))
            for j in new_end:
                end_indices[j] = min(end_indices[j], count)
            # if (prev_word[0] == form_manager.get_symbol_idx('<E>')) or (len(text_gen) >= checkpoint["opt"].dec_seq_length):
            #     break
            text_gen.append(prev_word)
            if end_indices.sum() < 1e8:
                break
            count += 1
        text_gen = torch.stack(text_gen, dim=1).tolist()
        text_gen = [[tg[:end_indices[j]]] for j, tg in enumerate(text_gen)]
        return text_gen, iters, expansions