import torch

def train(pairs_batch_train, pairs_batch_dev, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, num_epochs, device):
    clip = 1.0

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        for _, batch in enumerate(pairs_batch_train):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, pad_target_seqs_lengths = batch
            pad_input_seqs, pad_target_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device)
            
            train_loss = 0

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
        
            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)

            decoder_input = torch.ones(batch_size, 1).long().to(device) 
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
            
            attn_weights = torch.nn.functional.softmax(torch.ones(encoder_output.size(1), 1, encoder_output.size(0)), dim=-1).to(device)

            for i in range(0, pad_target_seqs.size(0)):
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, attn_weights)
                target = pad_target_seqs.squeeze()
                train_loss += criterion(decoder_output, target[i])
                decoder_input = pad_target_seqs[i]            

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
            encoder_optimizer.step()
            decoder_optimizer.step()



        # CALCULATE EVALUATION
        with torch.no_grad():
            for _, batch in enumerate(pairs_batch_dev):
                encoder.eval()
                decoder.eval()

                pad_input_seqs, input_seq_lengths, pad_target_seqs, pad_target_seqs_lengths = batch
                pad_input_seqs, pad_target_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device)

                dev_loss = 0

                encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
            
                decoder_input = torch.ones(batch_size, 1).long().to(device)
                decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
            
                attn_weights = torch.nn.functional.softmax(torch.ones(encoder_output.size(1), 1, encoder_output.size(0)), dim=-1).to(device)
            
                for i in range(0, pad_target_seqs.size(0)):     
                    decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output, attn_weights)
                    target = pad_target_seqs.squeeze()
                    dev_loss += criterion(decoder_output, target[i])
                    decoder_input = pad_target_seqs[i]

        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, train_loss.item(), dev_loss.item()))