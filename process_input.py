import json

def prepare_input_file(input_file_snli, input_file_multi, output_train, output_val):
    f_train = open(output_train, 'w')
    f_val = open(output_val, 'w')
    split = int(550152 * 0.98)
    multi_split = int(392702 * 0.97)
    print("SNLI split: ", split)
    print("Multi split: ", multi_split)

    with open(input_file_snli, 'r') as f_in:
        for idx, line in enumerate(f_in):
            data = eval(line)
            if idx < split:
                f_train.write(data['sentence1_processed'] + '\n')
                f_train.write(data['sentence2_processed'] + '\n')
            else:
                f_val.write(data['sentence1_processed'] + '\n')
                f_val.write(data['sentence2_processed'] + '\n')

    
    with open(input_file_multi, 'r') as f_in:
        for idx, line in enumerate(f_in):
            data = eval(line)
            if idx < multi_split:
                f_train.write(data['sentence1_processed'] + '\n')
                f_train.write(data['sentence2_processed'] + '\n')
            else:
                f_val.write(data['sentence1_processed'] + '\n')
                f_val.write(data['sentence2_processed'] + '\n')

    '''
    with open(input_multi_mismatched, 'r') as f_in:
        for line in f_in:
            data = eval(line)
            f_out.write(data['sentence1_processed'] + '\n')
            f_out.write(data['sentence2_processed'] + '\n')
    '''
    f_train.close()
    f_val.close() 
    

def process_text(text):
    text = text.replace('(', '').replace(')', '')
    return text.lower()

def process_snli(input_file, out_file):
    f_out = open(out_file, 'w')
    with open(input_file, 'r') as f_in:
        for line in f_in:
            data = eval(line)
            s1_parsed = process_text(data['sentence1_binary_parse']) 
            s2_parsed = process_text(data['sentence2_binary_parse'])
            data['sentence1_processed'] = s1_parsed
            data['sentence2_processed'] = s2_parsed
            json_str = json.dumps(data) + '\n' 
            f_out.write(json_str)
    f_out.close()

multi_nli_train = '../datasets/multinli_1.0/multinli_1.0_dev_mismatched.jsonl'
snli_train = '../datasets/snli_1.0/snli_1.0_train.jsonl'
snli_train_processed = '../datasets/snli_1.0/snli_1.0_train_processed.jsonl'
multi_nli_train_processed = '../datasets/multinli_1.0/multinli_1.0_train_processed.jsonl'

output_train = '../datasets/nli_data/train.txt'
output_val = '../datasets/nli_data/valid.txt'
#process_snli(snli_train, snli_train_processed)   
#print("SNLI done")
#process_snli(multi_nli_train, multi_nli_train_processed)
prepare_input_file(snli_train_processed, multi_nli_train_processed, output_train, output_val)
print("MultiNLI done")
