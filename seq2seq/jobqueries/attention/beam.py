import torch
import numpy as np

def beam(encoder, decoder, attention_decoder, enc_w_list, word_manager, form_manager, opt, using_gpu, checkpoint):
    iters = 0
    expansions = 0
    end_idx = form_manager.get_symbol_idx('<E>')
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
    # decode
    prev_h = enc_outputs[torch.arange(batch_size), end-lengths, :].unsqueeze(1).repeat(1, opt.beam_size, 1).to(device) # batch x beam x hidden
    prev_c = save_prev_c[torch.arange(batch_size), end-lengths, :].unsqueeze(1).repeat(1, opt.beam_size, 1).to(device) # batch x beam x hidden

    end_indices = torch.zeros(batch_size, opt.beam_size, dtype=torch.long).to(device) + 1e8
    text_gen = None
    prev_word = torch.zeros(batch_size, opt.beam_size, dtype=torch.long).to(device) + form_manager.get_symbol_idx('<S>')
    if opt.gpuid >= 0:
        prev_word = prev_word.cuda()
    count = 0
    log_probs = torch.cat([torch.zeros(batch_size, 1), torch.zeros(batch_size, opt.beam_size - 1) - 1e8], dim=1).to(device)

    while True:
        iters += 1
        expansions += ((log_probs > -1e8).long() * (end_indices == 1e8).long()).sum() # only count valid beams that aren't completed
        prev_c, prev_h = decoder(prev_word.flatten(), prev_c.flatten(0, 1), prev_h.flatten(0, 1))
        pred = attention_decoder(enc_outputs.unsqueeze(1).repeat(1, opt.beam_size, 1, 1).flatten(0, 1), prev_h, lengths.unsqueeze(1).repeat(1, opt.beam_size).flatten())
        prev_c, prev_h = prev_c.view(batch_size, opt.beam_size, -1), prev_h.view(batch_size, opt.beam_size, -1)
        pred = pred.view(batch_size, opt.beam_size, -1) # batch x beam x vocab of logprobs
        # make it so for the beams that are already ended, we keep only one copy
        end_found = (end_indices < 1e8).float() # batch x beam
        end_found_pad = torch.cat([torch.zeros(batch_size, opt.beam_size, 1), torch.zeros(batch_size, opt.beam_size, pred.shape[2] - 1) - 1e8], dim=2).to(device)
        current_log_probs = log_probs.unsqueeze(2) + (1 - end_found).unsqueeze(2) * pred + end_found.unsqueeze(2) * end_found_pad
        current_log_probs = current_log_probs.flatten(1) # batch x beam*vocab of logprobs accounting for past

        top_probs, top_indices = current_log_probs.topk(opt.beam_size, dim=1) # each batch x beam
        beam_indices = top_indices // pred.shape[2] # batch x beam
        vocab_indices = top_indices % pred.shape[2] # batch x beam
        log_probs = top_probs
        if text_gen is None:
            text_gen = vocab_indices.unsqueeze(2)
        else:
            text_gen = torch.gather(text_gen, 1, beam_indices.unsqueeze(2).expand_as(text_gen))
            text_gen = torch.cat([text_gen, vocab_indices.unsqueeze(2)], dim=2)
        end_indices = torch.gather(end_indices, 1, beam_indices).flatten()
        prev_c = torch.gather(prev_c, 1, beam_indices.unsqueeze(2).expand_as(prev_c))
        prev_h = torch.gather(prev_h, 1, beam_indices.unsqueeze(2).expand_as(prev_h))

        prev_word = vocab_indices
        new_end = (prev_word == end_idx).flatten().nonzero().flatten()
        if count >= checkpoint["opt"].dec_seq_length: # break out
            new_end = torch.arange(batch_size * opt.beam_size).to(device)
        end_indices[new_end] = torch.min(end_indices[new_end], torch.zeros_like(end_indices[new_end]) + count)
        end_indices = end_indices.view(batch_size, opt.beam_size)
        if end_indices.sum() < 1e8:
            break
        count += 1

    text_gen = text_gen.tolist()
    end_indices = end_indices.long()
    text_gen = [[text_gen[i][j][:end_indices[i][j]] for j in range(opt.beam_size)] for i in range(batch_size)]
    return text_gen, iters, expansions