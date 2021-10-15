"""
CS 593A
Assesment 4
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import RandomSampler
import torch
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

generated = tokenizer.encode("The year 2020 is")
context = torch.tensor([generated])

past = None

for i in range(5):
    print(i)
    output, past = model(context, past_key_values=past)
    #token = torch.argmax(output[..., -1, :])
    _,topk = torch.topk(output, k=50)
    token = RandomSampler(topk, replacement=True, num_samples=1)  

    generated += [token.tolist()]
    context = token.unsqueeze(0)

sequence = tokenizer.decode(generated)
print(sequence)
#samplek_output = model.generate(generated, do_sample=True, top_k=50)
#print(tokenizer.decode(samplek_output[0], skip_special_tokens=True))


#------------------------------------------------------------------------------

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
tf.random.set_seed(0)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
tf.random.set_seed(0)

input_ids = tokenizer.encode('The year 2020 is', return_tensors='tf')

# set top_k to 50
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
