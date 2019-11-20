# Deep LoFi Beats
Ivan Montero, Cameron Meissner, Anirudh Canumalla


## Main TODOs
- [x] Migrate code to github
- [ ] Make the code runnable on Windows/MacOS (ivan just copied over the code for the sake of existing)
- [ ] Write code to transform audio files to npy files, and save them
- [ ] Write seq2seq model
    - [ ] Implement data loader (Ivan)
    - [ ] Write training code (Cameron)
        - Assume data loader gives you two tensors:
            - prev (batch x seq_len_1)
            - next (batch x seq_len_2)
        - note that seq_len_1 =\\= seq_len_2
        - the seq2seq looks like the following: Given the previous subset of audio samples (where seq_len_1 is random), try to predict the following subset of audio samples (seq_len_2 of audio samples)
            - Visually: \[ ... | prev | next| ... \]
    - [ ] Write encoder decoder frameworks (Ani)
        - Encoder: an LSTM (stacked, unidirectional) that goes over the entirety of prev sequence. Output is the hidden state at the final 
        - Decoder: an LSTM (stacked, unidirectional) that has its initial hidden state instantiated with the result of the decoder. At each timestep, the output will be the size of the hidden layer. We will need a dense network to "transform" this output into a single float (which will correspond to the next "predicted" sample. Two ways of looking at this:
            - Training time: use "teacher forcing", e.g., at each timestep: given sample_t (input), predict sample_{t+1} (output). Later, we can modify this to randomly either use teacher forcing of greedy decoding (https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training-the-model)
            - Evaluation time: use "greedy decoding", e.g., use the output at timestep t (sample_t) as the input for timestep t+1 (sample_{t+1})
- [ ] Make seq2seq variational
    - [ ] Explore variation in an intermediate step (e.g, introduce "creativity" or "style change" in the middle of the audio)
- [ ] Salvage GAN
