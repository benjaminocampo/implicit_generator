import torch
import requests
from nltk.corpus import stopwords
from sentence_transformers import util

# Constants
REQUEST_TIMEOUT = 10  # seconds
BEARER_TOKEN_TEMPLATE = "Bearer {}"
CONTENT_TYPE_JSON = "application/json"
DECODE_ROUNDING_MODE = "trunc"
DEVICE = 'cpu'  # Assuming CPU, change to 'cuda' if GPU is used


# Helper function to sanitize prompts
def sanitize_prompt(prompts):
    return [prompt.replace("'", "").replace('"', "") for prompt in prompts]


# Helper function to construct headers for the API request
def construct_headers(secret_key):
    return {
        "Authorization": BEARER_TOKEN_TEMPLATE.format(secret_key),
        "Content-Type": CONTENT_TYPE_JSON
    }


# Language model querying function
def query_language_model(prompts,
                         secret_key,
                         model_endpoint,
                         topk=1,
                         max_tokens=1):
    sanitized_prompts = sanitize_prompt(prompts)
    payload = {
        "prompt": sanitized_prompts,
        "max_tokens": max_tokens,
        "temperature": 0.9,
        "n": 1,
        "stream": False,
        "logprobs": topk,
        "stop": ["<|endoftext|>", "\\n"]
    }

    try:
        response = requests.post(model_endpoint,
                                 headers=construct_headers(secret_key),
                                 json=payload,
                                 timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


def omit_unwanted_token(token, score, unwanted_tokens, prompts):
    if token not in unwanted_tokens and token.replace(
            ' ', '') not in unwanted_tokens and token in prompts:
        return -100
    return score


class BeamHypotheses(object):

    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp)**self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([
                    (s, idx) for idx, (s, _) in enumerate(self.beams)
                ])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def generate_imp_hs(prompt,
                    shots,
                    secret_key,
                    model_endpoint,
                    classifier,
                    sentence_sim,
                    lexicon,
                    end_token="\n",
                    weights=[.5, .5, .5, .5],
                    num_beams=10,
                    max_length=64,
                    length_penalty=1):
    """
    Generate sequences for each example using a modified beam search approach that 
    incorporates multiple scoring components, including language model probabilities, 
    semantic similarity, and classification logits.

    Parameters:
    - prompt (str): The initial text to start generating sequences from.
    - shots (list of str): Examples to guide the language model for task-specific generation.
    - secret_key (str): API key required for accessing the language model.
    - model_endpoint (str): The endpoint URL of the language model API.
    - classifier (callable): A pre-trained classifier model used to score sequences.
    - sentence_sim (callable): A semantic similarity model used to compute embeddings.
    - lexicon (DataFrame): A pandas DataFrame containing words and their associated scores.
    - end_token (str): Token that signifies the end of a sequence.
    - weights (list of float): The weights for each score component in the final score calculation.
    - num_beams (int): The number of beams to use in the beam search.
    - max_length (int): The maximum length of the sequence to generate.
    - length_penalty (float): Penalty for longer sequences in beam scoring.

    The function processes the input lexicon to remove stopwords and precomputes log values for lexicon entries.
    It encodes the provided examples (shots) to compute a mean embedding, which guides the semantic similarity
    during generation.

    The function initializes variables for the beam search, including hypothesis trackers, beam scores, and input IDs.
    It then enters a while loop that iterates until the maximum sequence length is reached or all beams are complete.

    Within each iteration of the while loop, the function:
    - Queries the language model with the current input IDs and retrieves log probabilities.
    - Computes and combines several score components, including language model scores, semantic similarity logits,
      classification logits, and lexicon logits.
    - Manages beam state updates, tracking hypotheses and pruning candidates.

    After the loop, the function finalizes all open beam hypotheses, ranking and retrieving the best hypothesis
    for each batch.

    Returns:
    - A list containing the best sequence generated for the prompt from each batch.
    """

    # Process lexicon to get dictionary of words to scores
    stops = stopwords.words('english')
    filtered_lexicon = lexicon[~lexicon["word"].isin(stops)]
    dict_lexicon = filtered_lexicon.set_index("word")["scaled_implicit_score"].to_dict()

    # Precompute the log values for the lexicon entries
    # This is assuming dict_lexicon values are scores that need to be converted to logits.
    log_dict_lexicon = {word: torch.log(torch.tensor(prob)).item() for word, prob in dict_lexicon.items()}

    # Embeddings for provided shots
    emb_shots = torch.mean(
        torch.stack([sentence_sim.encode(shot, convert_to_tensor=True) for shot in shots]),
        dim=0
    )

    # Initialize variables for beam search
    generated_hyps = [BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False)]
    beam_scores = torch.zeros((1, num_beams), dtype=torch.float, device=DEVICE)
    beam_scores[:, 1:] = float('-inf')  # Ensure only tokens from the first beam are considered

    # Preparing for the beam search
    done = [False]
    input_ids = [prompt] * num_beams
    step = 1
    start_index = len(prompt.split(' '))
    vocab_size=5
    pad_token_id = '<|pad|>'
    eos_token_ids = [end_token]

    #import pdb; pdb.set_trace()
    outputs = query_language_model(input_ids, secret_key, model_endpoint, topk=num_beams)
    outputs = outputs['choices'][0]['logprobs']['top_logprobs'][0]

    for i, (token, score) in enumerate(outputs.items()):
        beam_scores[0][i] = score
        input_ids[i] += token

    beam_scores = beam_scores.view(-1)    
    
    while step < max_length:
        # Query the language model with the given parameters.
        outputs = query_language_model(input_ids, secret_key, model_endpoint, topk=vocab_size)

        # Extract the scores from the outputs, which are assumed to be log probabilities.
        logprobs = [output['logprobs']['top_logprobs'] for output in outputs['choices'][:num_beams]]

        # Generate full_names and scores lists
        full_names = []
        scores = []
        for logprob in logprobs:
            keys = list(logprob[0].keys())
            values = list(logprob[0].values())
            filtered_scores = [omit_unwanted_token(keys[j], values[j], stops, prompt) for j in range(len(values))]
            full_names.extend(keys)
            scores.append(filtered_scores)

        # Convert the scores list to a PyTorch tensor and reshape it appropriately.
        scores_tensor = torch.tensor(scores).view(num_beams, -1)

        # Compute the next scores by adding beam scores and reshape as needed.
        next_scores = scores_tensor + beam_scores.unsqueeze(1)
        next_scores = next_scores.view(-1, vocab_size)

        # Select the top-k scores and tokens using torch.topk.
        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        next_scores = next_scores.view(-1)
        next_tokens = next_tokens.view(-1)

        # next_tokens is a tensor with shape [1, num_beams * 2] resulting from the torch.topk call
        # and full_names is a flattened list of all token names.

        # Use PyTorch's ability to index using a tensor to extract token names for the next tokens.
        next_tokens_names = [full_names[token_index] for token_index in next_tokens.tolist()]

        # Build the list of logits using list comprehension and look-up in the precomputed log_dict_lexicon.
        logits_lex = [log_dict_lexicon.get(w.strip(), 0.0) for w in next_tokens_names]

        # Convert the list of logits to a PyTorch tensor.
        logits_lex = torch.tensor(logits_lex)

        # First, construct the input sentences for the encoder in a list comprehension.
        input_sentences = [
            ' '.join(input_ids[t // vocab_size].split(' ')[start_index:]) + full_names[t % vocab_size]
            for t in next_tokens.view(-1).tolist()  # Flatten the next_tokens tensor to iterate over it.
        ]

        # Encode all sentences at once using batch processing.
        emb_text = sentence_sim.encode(input_sentences, convert_to_tensor=True)

        # Compute cosine similarity in a batch operation.
        cos_sim = util.pytorch_cos_sim(emb_shots, emb_text)  # Assumes both are tensors.

        # Take the log of cosine similarities.
        logits_sim = torch.log(cos_sim).view(-1)  # Flatten the tensor to a 1D tensor if needed.

        # Run the classifier on all input texts at once using batch processing.
        classifier_outputs = classifier(input_sentences, return_all_scores=True)

        # Extracting logits for the specified class (here index 2 is used as a placeholder).
        logits_clf = torch.log(1 - torch.nn.functional.softmax(torch.tensor([output[2]["score"] for output in classifier_outputs]), dim=0))

        # Perform the weighted sum of the scores on the device to prevent unnecessary data transfer.
        next_scores = (next_scores * weights[0] + 
                       logits_clf * weights[1] +
                       logits_lex * weights[2] + 
                       logits_sim * weights[3])


        # next batch beam content
        # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
        # Pre-calculate whether each batch is done
        batch_done_conditions = [generated_hyps[batch_idx].is_done(next_scores[batch_idx].max().item()) for batch_idx in range(1)]
        done = [done[batch_idx] or condition for batch_idx, condition in enumerate(batch_done_conditions)]

        # Check the preconditions for eos_token_ids and pad_token_id
        assert eos_token_ids is not None and pad_token_id is not None, "eos_token_id and pad_token have to be defined"

        next_batch_beam = []
        
        # for each sentence
        for batch_idx in range(1):
            # if we are done with this sentence
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item()
            )
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_ids is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            # check_ = [next_tokens[batch_idx][i] // vocab_size for i in range(len(next_tokens[batch_idx]))]
            # check_ = [torch.div(i, vocab_size, rounding_mode="trunc") for i in next_tokens[batch_idx]]
            for idx, score in zip(next_tokens, next_scores):
                # get beam and word IDs
                # beam_id = idx // vocab_size
                beam_id = torch.div(idx, vocab_size, rounding_mode="trunc")
                token_id = full_names[idx]
                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if eos_token_ids is not None and token_id in eos_token_ids:
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id], score.item(),
                    )
                else:
                    # add next predicted word if it is not eos_token
                    next_sent_beam.append((score, token_id, effective_beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == 1 * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = [x[1] for x in next_batch_beam]
        beam_idx = [x[2] for x in next_batch_beam]
        # re-order batch
        input_ids = [input_ids[i] for i in beam_idx]
        input_ids = [input_ids[i] + beam_tokens[i] for i in range(len(input_ids))]

        # stop when we are done with each sentence
        if all(done):
            break

        # update current length
        step = step + 1

    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(1):
        if done[batch_idx]:
            continue

        # add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)
    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch

    best_all = []
    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        best_all.append(sorted(hypotheses.beams, key=lambda x: x[0],reverse=True))
    return [p[-1] for p in best_all[0]]

