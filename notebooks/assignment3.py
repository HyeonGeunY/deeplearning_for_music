def rnn_single_step(
    current_input: torch.Tensor,
    prev_hidden: torch.Tensor,
    hh_weight: torch.Tensor,
    ih_weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    This function 
    
    Arguments:
        current_input: Input vector of the current time step. Has a shape of [input_dimension]
        prev_hidden: Hidden state from the previous time step. Has a shape of [hidden_dimension]
        hh_weight: Weight matrix for from hidden state to hidden state. Has a shape of [hidden_dimension, hidden_dimension]
        ih_weight: Weight matrix for from current input to hidden state. Has a shape of [input_dimension, hidden_dimension]
        bias: Bias of RNN. Has a shape of [hidden_dimension]
    
    Outputs:
        
    
    TODO: Complete this function
    """

    return torch.tanh(current_input @ ih_weight.T + prev_hidden @ hh_weight.T + bias)


def initialize_hidden_state_for_single_batch(hidden_dim: int) -> torch.Tensor:
    """
    This function returns zero Tensor for a given hidden dimension. This function assumes that the RNN uses single layer and single direction.
    
    Argument
        hidden_dim
      
    Return
         initial_hidden_state: Has a shape of [hidden_dim]
    
    TODO: Complete this function
    """
    return torch.zeros(hidden_dim)


def rnn_for_entire_timestep(
    input_seq: torch.Tensor,
    prev_hidden: torch.Tensor,
    hh_weight: torch.Tensor,
    ih_weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple:
    """
    This function returns the output of RNN for the given 'input_seq', for the given RNN's parameters (hh_weight, ih_weight, and bias)
    
    Arguments:
        input_seq: Sequence of input vector. Has a shape of [number_of_timestep, input_dimension]
        prev_hidden: Hidden state from the previous time step. Has a shape of [hidden_dimension]
        hh_weight: Weight matrix for from hidden state to hidden state. Has a shape of [hidden_dimension, hidden_dimension]
        ih_weight: Weight matrix for from current input to hidden state. Has a shape of [input_dimension, hidden_dimension]

    
    Return: tuple (output, final_hidden_state)
        output (torch.Tensor): Sequence of output hidden state of RNN along input timesteps. Has a a shape of [number_of_timestep, hidden_dimension]
        final_hidden_state (torch.Tensor): Hidden state of RNN of the last time step. Has a a shape of [hidden_dimension]
      
    TODO: Complete this function using your 'rnn_single_step()'
    """
    output = []
    for i, input in enumerate(input_seq):
        prev_hidden = torch.tanh(input @ ih_weight.T + prev_hidden @ hh_weight.T + bias)
        output.append(prev_hidden)
    output = torch.stack(output)
    return (output, prev_hidden)


class CustomEmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = torch.randn(num_embeddings, embedding_dim)

    def forward(self, x: torch.LongTensor):
        """
        Argument
          x: torch.LongTensor of arbitrary shape, where each element represent categorical index smaller than self.num_embeddings
          
        Return
          out (torch.Tensor): torch.FloatTensor with [shape of x, self.embedding_dim]
        
        TODO: Complete this function using self.weight
        """

        li_1 = []
        for i in x:
            li_2 = []
            for j in i:
                li_3 = []
                for k in j:
                    li_3.append(self.weight[k])
                li_3 = torch.stack(li_3)
                li_2.append(li_3)
            li_2 = torch.stack(li_2)
            li_1.append(li_2)
        li_1 = torch.stack(li_1)
        return li_1


class MelodyDataset:
    def __init__(self, muspy_dataset, vocabs=None):
        self.dataset = muspy_dataset

        if vocabs is None:
            self.idx2pitch, self.idx2dur = self._get_vocab_info()
            self.idx2pitch += ["start", "end"]
            self.idx2dur += ["start", "end"]
            self.pitch2idx = {x: i for i, x in enumerate(self.idx2pitch)}
            self.dur2idx = {x: i for i, x in enumerate(self.idx2dur)}

        else:
            self.idx2pitch, self.idx2dur, self.pitch2idx, self.dur2idx = vocabs

    def _get_vocab_info(self):
        entire_pitch = []
        entire_dur = []
        for note_rep in self.dataset:
            pitch_in_piece = note_rep[:, 1]
            dur_in_piece = note_rep[:, 2]
            entire_pitch += pitch_in_piece.tolist()
            entire_dur += dur_in_piece.tolist()
        return list(set(entire_pitch)), list(set(entire_dur))

    def get_vocabs(self):
        return self.idx2pitch, self.idx2dur, self.pitch2idx, self.dur2idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        This dataset class returns melody information as a tensor with shape of [num_notes, 2 (pitch, duration)].
        
        To train a melody language model, you have to provide a sequence of original note, and a sequence of next note for given original note.
        In other word, melody[i+1] has to be the shifted_melody[i], so that melody[i]'s next note can be retrieved by shifted_melody[i]
        (Remember, language model is trained to predict the next upcoming word)
        
        Also, to make genration easier, we usually add 'start' token at the beginning of sequence, and 'end' token at the end of the sequence.
        With these tokens, we can make the model recognize where is the start and end of the sequence explicitly.
        
        You have to add these tokens to the note sequence at this step.
        
        Argument:
          idx (int): Index of data sample in the dataset
        
        Returns:
          melody (torch.LongTensor): Sequence of [categorical_index_of_pitch, categorical_index_of_duration]
                                    Has a shape of [1 (start_token) + num_notes, 2 (pitch, dur)]. 
                                    The first element of the sequence has to be the index for 'start' token for both pitch and duration.
                                    The melody should not include 'end' token (Because we don't have to predict next note if we know that current note is 'end' token)
          shifted_melody (torch.LongTensor): Sequence of [categorical_index_of_pitch, categorical_index_of_duration]
                                            Has a shape of [num_notes + 1 (end_token), 2 (pitch, dur)]
                                            The i'th note of shifted melody has to be the same with (i+1)'th note of melody
                                            The shifted melody should not include 'start' token 
                                            (Because we never get a 'start' token after a note)

        TODO: Complete this function
        """
        # print(torch.tensor([self.pitch2idx['start'], self.dur2idx['start']]).unsqueeze(0))
        # print(self.dataset[idx][:-1, [1, 2]])

        melody_list = self.dataset[idx][:, [1, 2]]
        melody_list = [[self.pitch2idx[p], self.dur2idx[d]] for p, d in melody_list]

        melody = torch.cat(
            [
                torch.tensor([self.pitch2idx["start"], self.dur2idx["start"]]).unsqueeze(0),
                torch.tensor(melody_list),
            ],
            dim=0,
        ).type(torch.long)
        shifted_melody = torch.cat(
            [
                torch.tensor(melody_list),
                torch.tensor([self.pitch2idx["end"], self.dur2idx["end"]]).unsqueeze(0),
            ],
            dim=0,
        ).type(torch.long)

        return melody, shifted_melody


def your_collate_function(raw_batch):
    """
    You can make your own function to handle the batch
    """

    return raw_batch[0]


def pack_collate(raw_batch: list):
    """
    This function takes a list of data, and returns two PackedSequences
    
    Argument
        raw_batch: A list of MelodyDataset[idx]. Each item in the list is a tuple of (melody, shifted_melody)
                melody and shifted_melody has a shape of [num_notes (+1 if you don't consider "start" and "end" token as note), 2]
    Returns
        packed_melody (torch.nn.utils.rnn.PackedSequence)
        packed_shifted_melody (torch.nn.utils.rnn.PackedSequence)

    TODO: Complete this function
    """

    melody = []
    shifted_melody = []
    for i in range(len(raw_batch)):
        melody.append(raw_batch[i][0])
        shifted_melody.append(raw_batch[i][1])

    packed_melody = pack_sequence(melody, enforce_sorted=False)
    packed_shifted_melody = pack_sequence(shifted_melody, enforce_sorted=False)

    return packed_melody, packed_shifted_melody


class MelodyLanguageModel(nn.Module):
    def __init__(self, hidden_size, embed_size, vocabs):
        super().__init__()

        self.idx2pitch, self.idx2dur, self.pitch2idx, self.dur2idx = vocabs
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_pitch = len(self.idx2pitch)
        self.num_dur = len(self.idx2dur)
        self.num_layers = 3
        self.softmax = Softmax(dim=-1)

        """
        TODO: Declare four modules. Please follow the name strictly.
            1) self.pitch_embedder: nn.Embedding layer that embed pitch category index to a vector with size of 'embed_size'
            2) self.dur_embedder = nn.Embedding layer that embed duration category index to a vector with size of 'embed_size'
            3) self.rnn = nn.GRU layer that takes concatenated_embedding and has a hidden size of 'hidden_size', num_layers of self.num_layers, and batch_first=True
            4) self.final_layer = nn.Linear layer that takes self.rnn's output and convert it to logits (that can be used as input of softmax) of pitch + duration
        """

        self.pitch_embedder = Embedding(num_embeddings=self.num_pitch, embedding_dim=embed_size)
        self.dur_embedder = Embedding(num_embeddings=self.num_dur, embedding_dim=embed_size)
        self.rnn = GRU(2 * embed_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.final_layer = Linear(hidden_size, self.num_pitch + self.num_dur)
        self.prev_hidden = None

    def get_concat_embedding(self, input_seq):
        """
        This function returns concatenated pitch embedding and duration embedding for a given input seq
        
        Arguments:
            input_seq: A batch of melodies represented as a sequence of vector (pitch_idx, dur_idx). 
                        Has a shape of [num_batch, num_timesteps (num_notes), 2(pitch, dur)], or [num_timesteps (num_notes), 2]
                        벡터 (pitch_idx, dur_idx)의 시퀀스로 표현된 멜로디들의 집합으로 이루어진 배치. 
                        Shape은 [배치 샘플 수, 타임스텝의 수 (==음표의 수), 2 (음고, 길이)] 혹은 [타임스텝의 수 (num_notes), 2]
        Return:
             concat_embedding: A batch of sequence of concatenated embedding of pitch embedding and duration embedding.
                            Has a shape of [num_batch, num_timesteps (num_notes), embedding_size * 2]
                            Each vector of time t is [pitch_embedding ; duration_embedding] (concatenation)
                            
                            pitch embedding is the output of an nn.Embedding layer of given note pitch index
                            duration embedding is the output of an nn.Embedding layer of given note duration index
        
        TODO: Complete this function using self.pitch_embedder and self.dur_embedder
        You can use torch.cat to concatenate two tensors or vectors
        """

        if input_seq.dim() == 2:
            # input_seq = input_seq.unsqueeze(0)
            pitches = input_seq[:, 0]
            durs = input_seq[:, 1]
        if input_seq.dim() == 3:
            pitches = input_seq[:, :, 0]
            durs = input_seq[:, :, 1]
        # print(f"input_dim : {input_seq.dim()}")

        # print(pitches.unique())
        # print(durs.unique())
        pitches = self.pitch_embedder(pitches)
        durs = self.dur_embedder(durs)

        concat_embedding = torch.cat((pitches, durs), dim=-1)
        # print(f"concat_embedding_shape: {concat_embedding.shape}")
        return concat_embedding

    def initialize_rnn(self, batch_size: int) -> torch.Tensor:
        """
        This function returns initial hidden state for self.rnn for given batch_size
        
        Argument
          batch_size (int): 
          
        Return
          initial_hidden_state (torch.Tensor):
        """

        return torch.zeros([self.num_layers, batch_size, self.hidden_size])

    def forward(self, input_seq: torch.LongTensor):
        """
        Forward propgation of Melody Language Model.
        
        Argument
            input_seq: A batch of melodies represented as a sequence of vector (pitch_idx, dur_idx). 
                        Has a shape of [num_batch, num_timesteps (num_notes), 2(pitch, dur)], or can be a PackedSequence
                        벡터 (pitch_idx, dur_idx)의 시퀀스로 표현된 멜로디들의 집합으로 이루어진 배치. 
                        Shape은 [배치 샘플 수, 타임스텝의 수 (==음표의 수), 2 (음고, 길이)] 혹은 PackedSequence.
        
        Output
            pitch_dist: Probability distribution of pitch of next upcoming note for each timestep 't'.
                        Has a shape of [num_batch, numtimesteps, self.num_pitch]
                        매 타임 스텝 t에 대해, 그 다음에 등장할 음표 음고의 확률 분포
            dur_dist: Probability distribution of duration of next upcoming note for each timestep 't'.
                        Has a shape of [num_batch, numtimesteps, self.num_dur]
                        매 타임 스텝 t에 대해, 그 다음에 등장할 음표 길이의 확률 분포

        TODO: Complete this function. You have to handle both cases: input_seq as ordinary Tensor / input_seq as PackedSequence
        If the input_seq is PackedSequence, return PackedSequence
        
        
        input_seq → self.get_concat_embedding → self.rnn → self.final_layer → torch.softmax for [pitch, duration]
        
        Follow the instruction
        """

        if isinstance(input_seq, torch.Tensor):  # If input is an ordinary tensor

            # 1. Get concatenated_embeddings using self.get_concat_embedding

            concat_embedding = self.get_concat_embedding(input_seq).to(
                next(self.parameters()).device
            )  # .to(input_seq.device)

            # 2. Put concatenated_embeddings to self.rnn.
            # Remember: RNN, GRU, LSTM returns two outputs

            if not (self.prev_hidden == None):
                # h0 = self.initialize_rnn(input_seq.shape[0]).to(input_seq.device)
                self.prev_hidden = self.initialize_rnn(input_seq.shape[0]).to(
                    next(self.parameters()).device
                )
            output, self.prev_hidden = self.rnn(concat_embedding, self.prev_hidden)

            # 3. Put rnn's output with a shape of [num_batch, num_timestep, hidden_size] to self.final_layer
            output = self.final_layer(output)

            # 4. Convert logits (output of self.final_layer) to pitch probability and duration probability
            # Caution! You have to get separately softmax-ed pitch and duration
            # Because you have to pick one pitch and one duration from the probability distribution

            pitch_dist, dur_dist = (
                self.softmax(output[:, :, : self.num_pitch]),
                self.softmax(output[:, :, self.num_pitch :]),
            )

        elif isinstance(input_seq, PackedSequence):
            # 1. Get concatenated_embeddings using self.get_concat_embedding
            # To get concatenated_embeddings, You have to either pad_packed_sequence(input_seq, batch_first=True)
            # Or use input_seq.data, and then make new PackedSequence using concatenated_embeddings as data, and copy batch_lengths, sorted_indices, unsorted_indices.
            concat_embedding = PackedSequence(
                self.get_concat_embedding(input_seq.data),
                input_seq.batch_sizes,
                input_seq.sorted_indices,
                input_seq.unsorted_indices,
            ).to(input_seq.data.device)
            # print(f"concat_embedding: {concat_embedding.data.device}")
            # 2. Put concatenated embedding to self.rnn

            if not (self.prev_hidden == None):
                # h0 = self.initialize_rnn(input_seq.data.shape[0]).to(input_seq.data.device)
                self.prev_hidden = self.initialize_rnn(input_seq.data.shape[0]).to(
                    next(self.parameters()).device
                )

            _output, self.prev_hidden = self.rnn(concat_embedding, self.prev_hidden)

            # 3. Put rnn output to self.final_layer to get probability logit for pitch and duration
            # Again, rnn's output is PackedSequence so you have to handle it
            output = self.final_layer(_output.data)

            # 4. Convert logits to pitch probability and duration probability
            # Caution! You have to get separately softmax-ed pitch and duration
            # Because you have to pick one pitch and one duration from the probability distribution

            pitch_dist, dur_dist = (
                self.softmax(output[:, : self.num_pitch]),
                self.softmax(output[:, self.num_pitch :]),
            )
            pitch_dist = PackedSequence(
                pitch_dist, _output.batch_sizes, _output.sorted_indices, _output.unsorted_indices
            )
            dur_dist = PackedSequence(
                dur_dist, _output.batch_sizes, _output.sorted_indices, _output.unsorted_indices
            )

            # Return output as PackedSequence
        else:
            print(f"Unrecognized input type: {type(input_seq)}")

        return pitch_dist, dur_dist


def get_cross_entropy_loss(prob_distribution, correct_class):
    """
    This function takes predicted probability distrubtion and the corresponding correct_class.
    
    For example,  prob_distribution = [[0.2287, 0.2227, 0.5487], [0.1301, 0.4690, 0.4010]] means that
    for 0th data sample, the predicted probability for 0th category is 0.2287, for 1st category is 0.2227, and for 2nd category is 0.5487,
    and for 1st data sample, the predicted probability for 0th category is 0.1301, for 1st category is 0.4690, and for 2nd category is 0.4010,
    
    Cross entropy, which is -y*log(y_hat), can be regarded as negative log value of predicted probability for correct class (y==1).
    If the given correct_class is [1, 2], the loss for 0th data sample becomes negative log of [0.2287, 0.2227, 0.5487][1], which is -torch.log(0.2227), 
    because the correct category for this sample was 1st category, and the predicted probability was 0.2227
    And the loss for 1st data sample becomes negative log of [0.1301, 0.4690, 0.4010][2], which is -torch.log(0.4010),
    because the correct category for this sample was 2nd category, and the predicted probability was 0.4010
    
    To make implementation easy, let's assume we have 2D tensor for prob_distribution and  1D tensor for correct_class
    
    Arguments:
      prob_distribution (2D Tensor)
      correct_class (1D Tensor)
      
    Return:
      loss (torch.Tensor): Cross entropy loss for every data sample in prob_distrubition. Has a same shape with correct_class
    
    TODO: Complete this function
    
    Caution: When use torch.log(), don't forget to add small epsilon value (like 1e-6) to avoid infinity
    Do not return the mean loss. Return loss that has same shape with correct_class
    Try not to use for loop, or torch.nn.CrossEntropyLoss, or torch.nn.NLLLoss
    """
    assert (
        prob_distribution.dim() == 2 and correct_class.dim() == 1
    ), "Let's assume we only take 2D tensor for prob_distribution and 1D tensor for correct_class"
    # Write your code from here
    epsilon = 1e-7

    results = -1 * torch.log(
        prob_distribution[range(len(prob_distribution)), correct_class] + epsilon
    )

    return results


def get_loss_for_single_batch(model, batch, device):
    """
    This function takes model and batch and calculate Cross Entropy Loss for given batch.
    
    Arguments:
        model (MelodyLanguageModel)
        batch (batch collated by pack_collate): Tuple of (melody_batch, shifted_melody_batch)
        device (str): cuda or cpu. In which device to calculate the batch
        
    Return:
        loss (torch.Tensor): Calculated mean loss for given model and batch
      
    TODO: Complete this function using get_cross_entropy_loss().
    Now you have to return the mean loss of every data sample in the batch 
    
    Caution: You have to calculate loss for pitch, and loss for duration separately.
    Then you can take average of pitch_loss and duration_loss
    
    Important Tip: If you are using PackedSequence, you can feed PackedSequence.data directly to get_cross_entropy_loss.
    It makes the implementation much easier, because it doesn't need to reshape probabilty distribution to 2D and correct class to 1D.
    """

    pitch_dist, dur_dist = model(batch[0].to(device))
    pitch_loss = get_cross_entropy_loss(pitch_dist.data, batch[1].data[:, 0].to(device))
    dur_loss = get_cross_entropy_loss(dur_dist.data, batch[1].data[:, 1].to(device))

    return torch.mean(pitch_loss + dur_loss)


def get_initial_input_and_hidden_state(model, batch_size=1):
    """
    This function generates initial input vector and hidden state for model's GRU
    
    To generate a new sequence, you have to provide initial seed token, which is ['start', 'start'].
    You have to make a initial vector that has [pitch_category_index_of_'start', duration_category_index_of_'start']
    
    You also have to initial hidden state for the model's RNN.
    In uni-directional RNN(or GRU), hidden state of RNN has to be a zero tensor with shape of (num_layers, batch_size, hidden_size)

    
    Argument:
      model (MelodyLanguageModel)

    Returns:
      initial_input_vec (torch.Tensor): Has a shape of [batch_size, 1 (timestep), 2]
      initial_hidden (torch.Tensor): Has a shape of [num_layers, bach_size, hidden_size]
      
    TODO: Complete this function
    """
    initial_input_vec = torch.tensor([[[model.pitch2idx["start"], model.dur2idx["start"]]]])
    initial_input_vec = initial_input_vec.expand(batch_size, 1, 2)

    initial_hidden = model.initialize_rnn(batch_size)

    return initial_input_vec, initial_hidden


def predict_single_step(model, cur_input, prev_hidden):
    """
    This function runs MelodyLangaugeModel just for one step, for the given current input and previous hidden state.
    
    Arguments:
        model (MelodyLanguageModel)
        cur_input (torch.LongTensor): Input for the current time step. Has a shape of (batch_size=1, 1 (timestep), 2)
        prev_hidden (torch.Tensor): Hidden state of RNN after previous timestep

    Returns:
        cur_output (torch.LongTensor): Sampled note [pitch_category_idx, duration_category_idx] from the predicted probability distribution, with shape of [1,1,2]
        last_hidden (torch.Tensor): Hidden state of RNN
    Think about running the model.forward() step-by-step.
    
    input_seq → self.get_concat_embedding → self.rnn → self.final_layer → torch.softmax for [pitch, duration] → sampled [pitch, duration]

    """
    model.prev_hidden = prev_hidden
    pitch_dist, dur_dist = model(cur_input)
    pitch_out = pitch_dist.squeeze().multinomial(num_samples=1, replacement=True)
    dur_out = dur_dist.squeeze().multinomial(num_samples=1, replacement=True)
    cur_output = torch.LongTensor([[[pitch_out, dur_out]]])
    last_hidden = model.prev_hidden
    # print(f"cur_output: {cur_output}")
    # print(f"last_hidden: {last_hidden}")
    return cur_output, last_hidden


def is_end_token(model, cur_output):
    """
    During the generation, there is a possibility that the generated note predicted 'end' token for either pitch or duration.
    (In fact, model can even estimate 'start' token during the generation even though it has very low probability)
    
    Using information among (model.pitch2idx, model.dur2idx, model.idx2pitch, model.idx2dur, model.num_pitch, model.num_dur), check whether 
    
    Arguments:
      model (MelodyLanguageModel)
      cur_output (torch.LongTensor): Assume it has shape of [1,1,2 (pitch_idx, duration_idx)]
    
    Return:
      is_end_token (bool): True if cur_output include category index such as 'start' or 'end',
                            else False.
                            
    TODO: Complete this function
    """
    pitch_idx, dur_idx = cur_output.squeeze()
    pitch = model.idx2pitch[pitch_idx]
    dur = model.idx2dur[dur_idx]

    if pitch in ["start", "end"] or dur in ["start", "end"]:
        return True

    return False


def generate(model, random_seed=2022):
    """
    This function generates a new melody sequence with a given model and random_seed.
    
    Arguments:
      model (MelodyLanguageModel)
      random_seed (int): Language model's inference will always generate different result, because it uses random sampling for the prediction.
                        Therefore, if you want to reproduce the same generation result, you have to fix random_seed.
    
    Returns:
      generated_note_sequence (torch.LongTensor): Has a shape of [num_generated_notes, 2]
    
    TODO: Complete this function using get_initial_input_and_hidden_state(), predict_single_step(), is_end_token()
    
    Hint: You can use while loop
          You have to track the generated single note in a list or somewhere. 
    """

    torch.manual_seed(random_seed)  # To reproduce the result, we have to control random sequence

    """
    Write your code from here
    """
    generated_note_sequence = torch.LongTensor([])
    initial_vec, initial_hidden = get_initial_input_and_hidden_state(model, batch_size=1)
    cur_output, prev_hidden = predict_single_step(model, initial_vec, initial_hidden)

    while not is_end_token(model, cur_output):
        generated_note_sequence = torch.cat((generated_note_sequence, cur_output.squeeze(0)), dim=0)
        cur_output, prev_hidden = predict_single_step(model, cur_output, prev_hidden)

    return torch.LongTensor(generated_note_sequence)


def convert_idx_pred_to_origin(pred: torch.Tensor, idx2pitch: list, idx2dur: list):
    """
    This function convert neural net's output index to original pitch value (MIDI Pitch) and duration value 
    
    Argument:
    
        pred: generated output of the model. Has a shape of [num_notes, 2]. 
            0th dimension of each note represents pitch category index 
            and 1st dimension of each note represents duration category index
    
    Return:
        converted_out (torch.Tensor): Has a same shape with 'pred'.
        
    TODO: Complete this function
    """
    converted_out = torch.Tensor([[idx2pitch[p], idx2dur[d]] for p, d in pred])

    return converted_out


def convert_pitch_dur_to_note_representation(pitch_dur: torch.LongTensor):
    """
    This function takes pitch_dur (shape of [num_notes, 2]) and returns the corresponding note representation (shape of [num_notes, 4])
    In note representation, each note is represented as [start_timestep, pitch, duration, velocity]
    
    Since our generation is monophonic, you can regard start_timestep starts from 0 and accumulate the duration of note.
    You can fix velocity to 64.
    
  
    Arguments:
        pitch_dur: LongTensor of note where each note represented as pitch and duration value
        
    return:
        note_repr: numpy.Array with shape of [num_notes, 4]
                each note has value of [start_timestep, pitch, duration, velocity]

    TODO: Complete this function
    Hint: You can use torch.cumsum() to accumulate the duration.
    To convert torch tensor to numpy, you can use atensor.numpy()
    
    """

    start_timestep = torch.cat(
        (torch.tensor([0]), torch.cumsum(pitch_dur[:-1, 1], dim=0))
    ).unsqueeze(-1)
    velocity = torch.tensor([64] * len(pitch_dur)).unsqueeze(-1)
    note_repr = torch.cat((start_timestep, pitch_dur, velocity), dim=1)

    return note_repr.numpy().astype(int)


def generate_muspy_music(model, random_seed=0):
    """
    This function combines 'generate', 'convert_idx_pred_to_origin', 'convert_pitch_dur_to_note_representation', muspy.from_note_representation
    """
    gen_out = generate(model, random_seed)
    converted_out = convert_idx_pred_to_origin(gen_out, model.idx2pitch, model.idx2dur)
    note_repr = convert_pitch_dur_to_note_representation(converted_out)
    gen_music = muspy.from_note_representation(note_repr)
    return gen_music
