from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import h5py
import json
import sys
import retokenize

import torch
from pytorch_pretrained_bert import BertModel , BertTokenizer

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-data", nargs='+',    help="List of json data files",required=True)
parser.add_argument("-bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                         "bert-base-multilingual-cased, bert-base-chinese.")

parser.add_argument("-tokenizer", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                         "bert-base-multilingual-cased, bert-base-chinese.")

parser.add_argument("-outdir",  help="output directory",required=True)
parser.add_argument("-device",             help="dont use cuda when cpu", type=str, default="gpu")
parser.add_argument("-do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

params = parser.parse_args()

if params.device == "gpu":
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



def cache_dataset(data_path, out_file):
    with torch.no_grad():
        berttokenizer = BertTokenizer.from_pretrained(params.tokenizer, do_lower_case=params.do_lower_case)

        bertmodel = BertModel.from_pretrained(params.bert_model)

        bertmodel.eval()
        bertmodel.to(device)

    with open(data_path) as in_file:
        for doc_num, line in enumerate(in_file):
            example = json.loads(line)
            sentences = example["sentences"]
            file_key = example["doc_key"].replace("/", ":")
            group = out_file.create_group(file_key)
            concatenated_encoder_output_list = []
            for i, sentence in enumerate(sentences):
                encoder_output_by_layers = []
                encoder_output_torch = get_bert_embeddings(sentence,berttokenizer,bertmodel) # list of layer outputs
                for elem in encoder_output_torch:
                    if params.device == "gpu":
                        elem_np = elem.cpu().numpy()
                    else:
                        elem_np = elem.numpy()
                    encoder_output_by_layers.append(elem_np)
                encoder_output = np.transpose(np.asarray(encoder_output_by_layers), (1,2,0)) # num_layer x seq_len x dim -> seq_len x dim x num_layer
                group[str(i)] = encoder_output
            if doc_num % 10 == 0:
                print("Cached {} documents in {}".format(doc_num + 1, data_path))

def tokenize_data(berttokenizer, list_s1):
    s1 = " ".join(["[CLS]"] + list_s1 + ["[SEP]"])

    tokenized_text = berttokenizer.tokenize(s1)

    modified_tokenized_text = list(map(retokenize.process_bert_wordpiece_for_alignment, tokenized_text))
    bow_tokens = retokenize.space_tokenize_with_bow(s1.lower() if params.do_lower_case else s1)
    aligned = retokenize.align_lists(bow_tokens, modified_tokenized_text)
    aligned = [list(aligned[0].project_tokens(i)) for i, _ in enumerate(bow_tokens)]

    indexed_tokens = berttokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    return tokens_tensor, aligned #merge_index_list

def merge_embeddings_for_wpm_splits(encoded_layer, aligned_list):
    idx = 0
    final_list = []
    encoded_layer = encoded_layer[:,1:-1,:]
    aligned_list = aligned_list[1:-1]

    for elem in aligned_list:
        intermediate_list =[]
        for index in elem:
            intermediate_list.append(encoded_layer[:, index-1:index, :])
        if len(intermediate_list) >1:
            slice = torch.cat(intermediate_list,dim=1)
            final_list.append(torch.mean(slice, dim=1, keepdim=True))  # [batch, seq_length, hidden_size]
        elif len(intermediate_list) == 1:
            final_list.append(intermediate_list[0])  # [batch, seq_length, hidden_size]

    return torch.squeeze(torch.cat(final_list,dim = 1),dim = 0) # removing axis 0 which is batch axis = 1

def get_bert_embeddings(ip_list,berttokenizer,bertmodel):

    with torch.no_grad():
        tokens_tensor,aligned_list = tokenize_data(berttokenizer, ip_list)
        tokens_tensor = tokens_tensor.to(device)

        encoded_layers, _ = bertmodel(tokens_tensor)
        merged_encoded_layers = []
        for elem in encoded_layers:
            merged_encoded_layers.append(merge_embeddings_for_wpm_splits(elem,aligned_list))

        return merged_encoded_layers

if __name__ == "__main__":
    # token_ph, len_ph, lm_emb = build_elmo()
    for json_filename in params.data:
        model_name = params.bert_model.split("/")[-1]
        if not model_name:
                    model_name = params.bert_model.split("/")[-2] 
        out_file_name = params.outdir +model_name+"_"+ json_filename.split("/")[-1]+"_cache.hdf5"
        print(out_file_name+"\n")
        with h5py.File(out_file_name , "w") as out_file:
            cache_dataset(json_filename, out_file)

    print("data saved in: "+ out_file_name)
