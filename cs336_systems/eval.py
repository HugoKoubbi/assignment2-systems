import torch 
import torch.nn.functional as F
import timeit
import argparse
from cs336_basics.model import *
from cs336_basics.data import *
from cs336_basics.nn_utils import *
from cs336_basics.optimizer import *
from transformers import AutoTokenizer
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

if __name__ == 'main':
    parser=argparse.ArgumentParser()

    #Add all the hyperparameters in Parser mode
    #Training parameters
    parser.add_argument("--lr", default=2e-3,type=float)
    parser.add_argument("--wd", default=5e-3,type=float)
    parser.add_argument("--betas", default=(0.9, 0.999),type=tuple)
    parser.add_argument("--alpha_max", default=1e-3,type=float)
    parser.add_argument("--alpha_min", default=1e-4,type=float)
    parser.add_argument("--t_w",default=500,type=int) # choose 1-10% of the run that is in the warmup phase
    parser.add_argument("--t_c",default=5000,type=int) # choose such that the end of training coincides with the end of decay phases
    parser.add_argument("--max_norm",default=10.0,type=float)
    parser.add_argument("--context_length", default=256,type=int)
    parser.add_argument("--num_layers", default=4,type=int)
    parser.add_argument("--num_heads", default=16,type=int)
    parser.add_argument("--d_model", default=512,type=int)
    parser.add_argument("--d_ff", default=1280,type=int)
    parser.add_argument("--theta", default=10000,type=int)
    parser.add_argument("--vocab_size", default=8192,type=int)
    parser.add_argument('--iterations', default=5005,type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--number_tokens', default=400000000, type=int)
    parser.add_argument('--Device', default='mps')


config_GPT2_small={ 'lr': 2e-3, 'wd': 5e-3, 'betas': (0.9, 0.999), 'alpha_max': 1e-3, 'alpha_min': 1e-4, 't_w': 500, 't_c': 5000, 'max_norm': 10.0, 'context_length': 256, 'num_layers': 12, 'num_heads': 12, 'd_model': 768, 'd_ff': 3072, 'theta': 10000, 'vocab_size': 10000, 'iterations': 5005, 'batch_size': 32, 'number_tokens': 400000000}
config_GPT2_medium={ 'lr': 2e-3, 'wd': 5e-3, 'betas': (0.9, 0.999), 'alpha_max': 1e-3, 'alpha_min': 1e-4, 't_w': 500, 't_c': 5000, 'max_norm': 10.0, 'context_length': 256, 'num_layers': 24, 'num_heads': 16, 'd_model': 1024, 'd_ff': 4096, 'theta': 10000, 'vocab_size': 8192, 'iterations': 5005, 'batch_size': 32, 'number_tokens': 400000000}
config_GPT2_large={ 'lr': 2e-3, 'wd': 5e-3, 'betas': (0.9, 0.999), 'alpha_max': 1e-3, 'alpha_min': 1e-4, 't_w': 500, 't_c': 5000, 'max_norm': 10.0, 'context_length': 256, 'num_layers': 36, 'num_heads': 20, 'd_model': 1280, 'd_ff': 5120, 'theta': 10000, 'vocab_size': 8192, 'iterations': 5005, 'batch_size': 32, 'number_tokens': 400000000}
config_GPT2_xlarge={ 'lr': 2e-3, 'wd': 5e-3, 'betas': (0.9, 0.999), 'alpha_max': 1e-3, 'alpha_min': 1e-4, 't_w': 500, 't_c': 5000, 'max_norm': 10.0, 'context_length': 256, 'num_layers': 32, 'num_heads': 32, 'd_model': 2560, 'd_ff': 10240, 'theta': 10000, 'vocab_size': 8192, 'iterations': 5005, 'batch_size': 32, 'number_tokens': 400000000}
conig_GPT2_10B={ 'lr': 2e-3, 'wd': 5e-3, 'betas': (0.9, 0.999), 'alpha_max': 1e-3, 'alpha_min': 1e-4, 't_w': 500, 't_c': 5000, 'max_norm': 10.0, 'context_length': 256, 'num_layers': 50, 'num_heads': 36, 'd_model': 4608, 'd_ff': 12288, 'theta': 10000, 'vocab_size': 8192, 'iterations': 5005, 'batch_size' : 32, 'number_tokens': 400000000}


print(config_GPT2_small['d_ff'])

def benchmarking_script(config, w, n, mode,device=None):
    d_model=config['d_model']
    d_ff=config['d_ff']
    vocab_size=config['vocab_size']
    num_layers=config['num_layers']
    num_heads=config['num_heads']
    rope_theta=config['theta']
    context_length=config['context_length']
    batch_size=config['batch_size']
    lr= config['lr']
    betas=config['betas']
    wd=config['wd']
    max_norm=config['max_norm']
    #Load right Device
    if torch.backends.mps.is_available():
        Device = torch.device("mps")
    elif torch.cuda.is_available():
        Device= torch.device("cuda")
    else:    
        Device = torch.device("cpu")
    print("Using:", Device)

    model=BasicsTransformerLM(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta)
    model.to(Device)
    # Load data 
    with open('cs336-basics/data/TinyStoriesV2-GPT4-train.txt','r', encoding="utf-8") as training_data:
        with open('cs336-basics/data/TinyStoriesV2-GPT4-valid.txt','r', encoding="utf-8") as test_data:
            # Prepare the tokenizer 
            #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            #model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            #tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="original") 
            tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>'"))
            trainer = BpeTrainer(special_tokens=["<|endoftext|>'"],vocab_size=10000)
            tokenizer.pre_tokenizer = Whitespace()
            #files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
            #training_data = training_data.read(batch_size * context_length)
            file=['cs336-basics/data/TinyStoriesV2-GPT4-train.txt']
            tokenizer.train(file, trainer) 
            #tokenizer.train(files, trainer) 
            tokenizer.save("cs336_systems/tokenizer-wiki.json")
            tokenizer = Tokenizer.from_file("cs336_systems/tokenizer-wiki.json")



            #print(f'Vocabulary size: {len(tokenizer.vocab)}') 
            #print(f'Merges size: {len(tokenizer.merges)}')
            print(f'Tokenizer initialized.')
            training_data = training_data.read(batch_size * context_length)
            # Tokenize the data
            #training_data = training_data.read(batch_size * context_length)
            np.save('cs336-basics/data/training_tokenized' ,tokenizer.encode(training_data))
            #print(tokenizer.encode(training_data))
            #print(tokenizer.encode(training_data).ids)
            #print(tokenizer.encode(training_data).tokens)
            #np.save('cs336-basics/data/test_tokenized',tokenizer.encode(test_data))

    # Tokenize the data
    np.save('cs336-basics/data/training_tokenized.npy' ,tokenizer.encode(training_data).ids)
    #np.save('cs336-basics/data/test_tokenized',tokenizer.encode(test_data))
    print()
    training_tokenized_mm=np.load('cs336-basics/data/training_tokenized.npy',mmap_mode='r')

    if mode=='forward':
        with torch.no_grad():
            for step in range(w):
                inputs,output=get_batch(training_tokenized_mm,batch_size,context_length,Device)
                print(inputs.shape)
                print(inputs.device)
                model(inputs)
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete before
            time_base=timeit.timeit()
            for step in range(n):
                inputs,output=get_batch(training_tokenized_mm,batch_size,context_length,Device)
                model(inputs)
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete before
            time_final=timeit.timeit()
            return( time_final-time_base)
    elif mode=='forward_backward':
        inputs,output=get_batch(training_tokenized_mm,batch_size,context_length,Device)
        optimizer=AdamW(model.parameters(),lr=lr,betas=betas,weight_decay=wd)
        model.train()
        for w in range(w):
            optimizer.zero_grad()
            loss=cross_entropy(model(inputs),output)
            loss.backward()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete before timing
        time_taken=timeit.timeit()
        for i in range(n):
            optimizer.zero_grad()
            loss=cross_entropy(model(inputs),output)
            loss.backward()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete before timing
        time_final=timeit.timeit()
        return( time_final-time_taken)
    elif mode=='training':
        inputs,output=get_batch(training_tokenized_mm,batch_size,context_length,Device)
        optimizer=AdamW(model.parameters(),lr=lr,betas=betas,weight_decay=wd)
        model.train()
        for w in range(w):
            optimizer.zero_grad()
            loss=cross_entropy(model(inputs),output)
            loss.backward()
            clip_gradient(model.parameters(), max_norm)
            optimizer.step()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete before timing
        time_taken=timeit.timeit()
        for i in range(n):
            optimizer.zero_grad()
            loss=cross_entropy(model(inputs),output)
            loss.backward()
            clip_gradient(model.parameters(), max_norm)
            optimizer.step()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete before timing
        time_final=timeit.timeit()
        return( time_final-time_taken)

print(benchmarking_script(config_GPT2_small, 10, 100,'forward'))