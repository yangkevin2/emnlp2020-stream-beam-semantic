import torch
import numpy as np

def pad_to_max_length(t1, t2, dim, side='right'):
    if t1.size(dim) < t2.size(dim):
        t1 = pad_to_length(t1, t2.size(dim), dim, side)
    elif t2.size(dim) < t1.size(dim):
        t2 = pad_to_length(t2, t1.size(dim), dim, side)
    return t1, t2


def pad_to_length(tensor, length, dim, side='right', pad_value=0):
    assert side in ['left', 'right']
    assert tensor.size(dim) <= length
    if tensor.size(dim) == length:
        return tensor
    else:
        zeros_shape = list(tensor.shape)
        zeros_shape[dim] = length - tensor.size(dim)
        zeros_shape = tuple(zeros_shape)
        if side == 'right':
            return torch.cat([tensor, torch.zeros(zeros_shape).type(tensor.type()).to(tensor.device) + pad_value], dim=dim)
        else:
            return torch.cat([torch.zeros(zeros_shape).type(tensor.type()).to(tensor.device) + pad_value, tensor], dim=dim)


def select_source_indices(num_valid_beams, master_progress, max_beams, min_prog):
    # select source infos (starting from the least progress made) until we hit max allowed beams
    if min_prog:
        indices = torch.LongTensor(list(range(len(num_valid_beams)))).to(num_valid_beams.device)
        prog_min = master_progress.min()
        mp_indices = (master_progress == prog_min).nonzero().flatten()
        nvb = num_valid_beams[mp_indices]
        num_beams = torch.cumsum(nvb, dim=0)
        allowed_mask = (num_beams <= max_beams).float()
        selected_indices = mp_indices[:int(allowed_mask.sum())]
        unselected_mask = torch.ones_like(master_progress)
        unselected_mask[selected_indices] = 0
        unselected_indices = unselected_mask.nonzero().flatten()
        return selected_indices, unselected_indices
    else:
        indices = list(range(len(num_valid_beams)))
        nvb = num_valid_beams[indices]
        num_beams = torch.cumsum(nvb, dim=0)
        allowed_mask = (num_beams <= max_beams).float()
        return torch.LongTensor(indices[:int(allowed_mask.sum())]).to(num_valid_beams.device), \
            torch.LongTensor(indices[int(allowed_mask.sum()):]).to(num_valid_beams.device) # selected, unselected


def pad_mask(lengths: torch.LongTensor, device='cuda', max_seqlen=None) -> torch.ByteTensor:
    # lengths: bs. Ex: [2, 3, 1]
    if max_seqlen is None:
        max_seqlen = torch.max(lengths)
    expanded_lengths = lengths.unsqueeze(0).repeat((max_seqlen, 1))  # [[2, 3, 1], [2, 3, 1], [2, 3, 1]]
    indices = torch.arange(max_seqlen).unsqueeze(1).repeat((1, lengths.size(0))).to(device)  # [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    return expanded_lengths > indices  # pad locations are 0. #[[1, 1, 1], [1, 1, 0], [0, 1, 0]]. seqlen x bs


def variable_beam_stream(encoder, decoder, attention_decoder, enc_w_list, word_manager, form_manager, opt, using_gpu, checkpoint, min_prog=False):
    iters = 0
    expansions = 0
    find_top_z = opt.beam_size
    ap, mc, k = opt.ap, opt.mc, opt.beam_size
    encode_batch_size = opt.batch_size
    encode_batch_diff_limit = opt.batch_size * 100
    max_sources = opt.batch_size * 1000
    max_beams = opt.batch_size * opt.beam_size
    end_idx = form_manager.get_symbol_idx('<E>')
    start_idx = form_manager.get_symbol_idx('<S>')
    # initialize the rnn state to all zeros
    device = 'cuda:' + str(opt.gpuid) if using_gpu else 'cpu'
    total_length = len(enc_w_list)
    for ewl in enc_w_list:
        ewl.append(word_manager.get_symbol_idx('<S>'))
        ewl.insert(0, word_manager.get_symbol_idx('<E>'))

    current_idx = 0
    master_lengths = torch.zeros(0).long().to(device) # batch
    master_log_probs = torch.zeros(0, k).to(device) # batch x k
    master_prev_c = torch.zeros(0, k, encoder.hidden_size).to(device) # batch x beam x hidden
    master_prev_h = torch.zeros(0, k, encoder.hidden_size).to(device) # batch x beam x hidden
    master_enc_outputs = torch.zeros(0, 0, encoder.hidden_size).to(device) # batch x seq x hidden
    master_index = torch.zeros(0).long().to(device) # batch
    master_valid_beam_mask = torch.zeros(0, k).long().to(device)
    master_num_valid_beams = torch.zeros(0).long().to(device) # batch
    master_end_indices = torch.zeros(0, k).long().to(device) # batch x k
    master_text_gen = torch.zeros(0, k, 0).long().to(device) # batch x k x seq
    master_progress = torch.zeros(0).long().to(device) # batch
    master_done_lengths = torch.zeros(0).long().to(device) # batch
    master_first_done_log_prob = torch.zeros(0).to(device) - 1e8

    master_done_beams = [[] for _ in range(total_length)]

    while True:
        starting_new_encode_group = False
        while (len(master_index) <= max_sources - encode_batch_diff_limit or (starting_new_encode_group and len(master_index) <= max_sources - encode_batch_size)) and current_idx < total_length:
            starting_new_encode_group = True
            new_sources = enc_w_list[current_idx:current_idx + encode_batch_size]
            batch_size = len(new_sources)
            lengths = torch.tensor([len(ewl) for ewl in new_sources]).to(device)
            end = lengths.max()
            for i in range(len(new_sources)):
                if len(new_sources[i]) < end:
                    new_sources[i] = [0 for _ in range(end - len(new_sources[i]))] + new_sources[i]
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
            inputs = torch.tensor(new_sources, dtype=torch.long) # batch x seq
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
            prev_word = torch.zeros(batch_size, opt.beam_size, dtype=torch.long).to(device) + form_manager.get_symbol_idx('<S>')
            if opt.gpuid >= 0:
                prev_word = prev_word.cuda()
            # count = 0
            log_probs = torch.cat([torch.zeros(batch_size, 1), torch.zeros(batch_size, opt.beam_size - 1) - 1e8], dim=1).to(device)

            current_seqlen = max(1, master_text_gen.shape[2])
            enc_max_len = max(master_enc_outputs.shape[1], enc_outputs.shape[1])
            master_text_gen = torch.cat([pad_to_length(master_text_gen, current_seqlen, 2, side='left'), 
                                         pad_to_length(torch.zeros(batch_size, k, 1) + start_idx, current_seqlen, 2, side='left').long().to(device)],
                                        dim=0)
            master_lengths = torch.cat([master_lengths, lengths], dim=0)
            master_log_probs = torch.cat([master_log_probs, log_probs], dim=0)
            master_prev_c = torch.cat([master_prev_c, prev_c], dim=0)
            master_prev_h = torch.cat([master_prev_h, prev_h], dim=0)
            master_enc_outputs = torch.cat([pad_to_length(master_enc_outputs, enc_max_len, 1, side='left'), \
                                            pad_to_length(enc_outputs, enc_max_len, 1, side='left')], dim=0)
            master_index = torch.cat([master_index, torch.arange(batch_size).to(device) + current_idx], dim=0)
            master_valid_beam_mask = torch.cat([master_valid_beam_mask, torch.cat([torch.ones(batch_size, 1), torch.zeros(batch_size, k-1)], dim=1).to(device).long()], dim=0)
            master_num_valid_beams = torch.cat([master_num_valid_beams, torch.ones(batch_size).long().to(device)], dim=0)
            master_end_indices = torch.cat([master_end_indices, end_indices.long()], dim=0)
            master_progress = torch.cat([master_progress, torch.zeros(batch_size).long().to(device)], dim=0)
            master_done_lengths = torch.cat([master_done_lengths, torch.zeros(batch_size).long().to(device)], dim=0)
            master_first_done_log_prob = torch.cat([master_first_done_log_prob, torch.zeros(batch_size).to(device)-1e8], dim=0)

            current_idx += encode_batch_size

        selected_indices, unselected_indices = select_source_indices(master_num_valid_beams, master_progress, max_beams, min_prog)
        text_gen = master_text_gen[selected_indices]
        lengths = master_lengths[selected_indices]
        log_probs = master_log_probs[selected_indices]
        prev_c = master_prev_c[selected_indices]
        prev_h = master_prev_h[selected_indices]
        enc_outputs = master_enc_outputs[selected_indices]
        index = master_index[selected_indices]
        valid_beam_mask = master_valid_beam_mask[selected_indices]
        num_valid_beams = master_num_valid_beams[selected_indices]
        end_indices = master_end_indices[selected_indices]
        done_lengths = master_done_lengths[selected_indices]
        progress = master_progress[selected_indices]
        first_done_log_prob = master_first_done_log_prob[selected_indices]
        batch_size = len(index)

        iters += 1
        expansions += ((log_probs > -1e8).long() * (end_indices == 1e8).long()).sum() 
        # note num_valid_beams also includes lower-down beams that are completed, since we don't bother with that

        prev_word = text_gen[:, :, -1].flatten() # batch*k
        valid_beam_indices = valid_beam_mask.flatten().nonzero().flatten()
        prev_c, prev_h = decoder(prev_word.flatten()[valid_beam_indices], prev_c.flatten(0, 1)[valid_beam_indices], prev_h.flatten(0, 1)[valid_beam_indices])
        pred = attention_decoder(enc_outputs.unsqueeze(1).repeat(1, opt.beam_size, 1, 1).flatten(0, 1)[valid_beam_indices], prev_h, lengths.unsqueeze(1).repeat(1, opt.beam_size).flatten()[valid_beam_indices])
        reverse_idx = (torch.cumsum(valid_beam_mask.flatten(), dim=0) * valid_beam_mask.flatten()).long() - 1 # it's fine to select whatever position for padding as they'll be removed later
        prev_c = prev_c[reverse_idx].view(batch_size, k, -1)
        prev_h = prev_h[reverse_idx].view(batch_size, k, -1)
        pred = pred[reverse_idx].view(batch_size, k, -1)
        
        # make it so for the beams that are already ended, we keep only one copy
        end_found = (end_indices < 1e8).float() 
        end_found_pad = torch.cat([torch.zeros(batch_size, opt.beam_size, 1), torch.zeros(batch_size, opt.beam_size, pred.shape[2] - 1) - 1e8], dim=2).to(device)
        current_log_probs = log_probs.unsqueeze(2) + (1 - end_found).unsqueeze(2) * pred + end_found.unsqueeze(2) * end_found_pad
        # current_log_probs = current_log_probs.flatten(1) # batch x beam*vocab of logprobs accounting for past

        
        mc_probs, mc_indices = current_log_probs.topk(mc, dim=2) # batch x beam x mc
        top_log_probs, top_indices = mc_probs.flatten(1).topk(opt.beam_size, dim=1) # batch x beam
        mc_vocab_indices = top_indices % mc
        beam_indices = top_indices // mc # batch x beam
        vocab_indices = torch.gather(mc_indices.flatten(1), 1, (mc_vocab_indices + beam_indices*mc)) # batch x beam

        # check which vocab_indices are done (in the first beam position), and add the corresponding beam to an array of done predictions
        end_indices = torch.gather(end_indices, 1, beam_indices)
        newly_done_all = ((vocab_indices == end_idx).long() + (end_indices < 1e8).long()).clamp(max=1) # batch, k
        newly_done = torch.cumprod(newly_done_all, dim=1) # keep on beam if there's something above it that's not done yet
        newly_done_indices = newly_done.flatten().nonzero().flatten() # batch*k
        
        newly_done_beam_indices = newly_done_all.flatten().nonzero().flatten()
        end_indices = end_indices.flatten()
        end_indices[newly_done_beam_indices] = torch.min(end_indices[newly_done_beam_indices], progress.unsqueeze(1).repeat(1, k).flatten()[newly_done_beam_indices])
        end_indices = end_indices.view(batch_size, k)
        done_lengths += newly_done.sum(dim=1).flatten() # update this one before others since we'll need it earlier
        
        for i, j in enumerate(newly_done_indices): # TODO no for loop here
            source_idx = j // k
            # add to some master list with an entry for each source
            if len(master_done_beams[index[source_idx]]) < find_top_z:
                if len(master_done_beams[index[source_idx]]) == 0:
                    first_done_log_prob[source_idx] = top_log_probs[source_idx, j % k]
                master_done_beams[index[source_idx]].append( \
                        (text_gen[source_idx, beam_indices[source_idx, j % k]].flatten().tolist(), end_indices[source_idx, j % k], progress[source_idx], top_log_probs[source_idx, j % k]))
        
        # then, shift log_probs and beam_indices for those beams and delete that beam(s); put in placeholder beam and log_prob at the k^th position
        # need to shift top_log_probs, beam_indices, vocab_indices accordingly
        top_log_probs = torch.cat([top_log_probs, torch.zeros_like(top_log_probs).to(device) - 1e8], dim=1) # 1, batch, 2k
        shift_indices = newly_done.sum(dim=1).unsqueeze(1) + torch.arange(k).to(device).unsqueeze(0) # batch, k
        top_log_probs = torch.gather(top_log_probs, 1, shift_indices)
        shift_indices = shift_indices.clamp(max=k-1)
        beam_indices = torch.gather(beam_indices, 1, shift_indices)
        vocab_indices = torch.gather(vocab_indices, 1, shift_indices)
        newly_done_all = torch.gather(newly_done_all, 1, shift_indices)
        prev_c = torch.gather(prev_c, 1, beam_indices.unsqueeze(2).expand_as(prev_c))
        prev_h = torch.gather(prev_h, 1, beam_indices.unsqueeze(2).expand_as(prev_h))
        end_indices = torch.gather(end_indices, 1, shift_indices)
        
        log_probs = top_log_probs
        ap_thresholds = (torch.max(log_probs[:, 0], first_done_log_prob) - ap).unsqueeze(1) # batch x 1
        valid_beam_mask = (log_probs > ap_thresholds).long() # batch x k
        # update valid beam mask based on how many beams are left for each source
        done_mask = pad_mask(k - done_lengths, device, max_seqlen=k).permute(1, 0) # batch x k of beams to keep, up to k - num done already
        found_z_mask = (done_lengths >= find_top_z).unsqueeze(1)
        valid_beam_mask = valid_beam_mask * done_mask * (1-found_z_mask.long())

        log_probs = log_probs - 1e8 * (1-valid_beam_mask)

        new_master_indices = torch.zeros(master_text_gen.size(0), k).long().to(device) # batch x k
        new_master_indices[selected_indices] = vocab_indices
        master_text_gen[selected_indices] = torch.gather(master_text_gen[selected_indices], 1, beam_indices.unsqueeze(2).expand_as(master_text_gen[selected_indices]))
        master_text_gen = torch.cat([master_text_gen, new_master_indices.unsqueeze(2)], dim=2)
        master_text_gen = torch.cat([torch.zeros(master_text_gen.size(0), k, 1).long().to(device), master_text_gen], dim=2)
        master_text_gen[selected_indices] = torch.roll(master_text_gen[selected_indices], -1, 2)
        master_text_gen = master_text_gen[:, :, :-1]
        master_log_probs[selected_indices] = log_probs
        master_prev_c[selected_indices] = prev_c
        master_prev_h[selected_indices] = prev_h
        master_valid_beam_mask[selected_indices] = valid_beam_mask
        master_num_valid_beams[selected_indices] = valid_beam_mask.sum(dim=1)
        master_progress[selected_indices] = progress + 1
        master_done_lengths[selected_indices] = done_lengths
        master_end_indices[selected_indices] = end_indices
        master_first_done_log_prob[selected_indices] = first_done_log_prob

        # filter the done ones out of all the master tensors
        exceed_length_mask = (progress) >= checkpoint["opt"].dec_seq_length
        no_beams_mask = (valid_beam_mask.sum(dim=1) == 0)
        keep_indices = ((1 - exceed_length_mask.long()) * (1 - no_beams_mask.long()) * (1 - found_z_mask.long().flatten())).flatten().nonzero().flatten().long()
        keep_indices = torch.cat([selected_indices[keep_indices], unselected_indices], dim=0)
        exceed_length_indices = exceed_length_mask.long().nonzero().flatten()
        for source_idx in exceed_length_indices:
            master_done_beams[index[source_idx]].append( \
                        (text_gen[source_idx, beam_indices[source_idx, 0]].flatten().tolist(), progress[source_idx], progress[source_idx], top_log_probs[source_idx, 0]))
        
        master_lengths = master_lengths[keep_indices]
        master_log_probs = master_log_probs[keep_indices]
        master_prev_c = master_prev_c[keep_indices]
        master_prev_h = master_prev_h[keep_indices]
        master_enc_outputs = master_enc_outputs[keep_indices]
        master_index = master_index[keep_indices]
        master_valid_beam_mask = master_valid_beam_mask[keep_indices]
        master_num_valid_beams = master_num_valid_beams[keep_indices]
        master_end_indices = master_end_indices[keep_indices]
        master_text_gen = master_text_gen[keep_indices]
        master_progress = master_progress[keep_indices]
        master_done_lengths = master_done_lengths[keep_indices]
        master_first_done_log_prob = master_first_done_log_prob[keep_indices]

        # break if none left
        if len(master_index) == 0 and current_idx >= total_length:
            break

        # remove padding for text_gen and encoder_outputs
        if len(master_index) > 0:
            while master_text_gen[:, :, 0].sum() == 0:
                master_text_gen = master_text_gen[:, :, 1:]
            while enc_outputs[:, 0].sum() == 0:
                enc_outputs = enc_outputs[:, 1:]

    for i in range(len(master_done_beams)):
        master_done_beams[i] = [x[0][- x[2]: (-x[2] + x[1]) if (-x[2] + x[1]) < 0 else 1000] for x in master_done_beams[i]]            
    
    return master_done_beams, iters, expansions