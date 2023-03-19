from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch,json,os,sys
import pandas as pd
import numpy as np
from datasets import Dataset
from unzip_load import download_zip
from json_convert import transform_json
import ast
#model_checkpoint = "/content/drive/MyDrive/bert_large_trained_initial/checkpoint-3000"
#model_checkpoint = "bert-large-uncased-whole-word-masking-finetuned-squad"

from transformers import AutoTokenizer



def add_tokenised_features(dataset,max_length = 384,doc_stride = 128,tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')):
        
    tokenized_list = tokenizer(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
   
    sample_map = tokenized_list.pop("overflow_to_sample_mapping")
    offset_map = tokenized_list.pop("offset_mapping")

    start_pos = []
    end_pos = []

    for index, offset in enumerate(offset_map):
        input_ids = tokenized_list["input_ids"][index]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        seq_ids = tokenized_list.sequence_ids(index)

        answers = dataset["answer"][sample_map[index]]

        if len(answers["answer_start"]) == 0:
            start_pos.append(cls_index)
            end_pos.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            word_start_char_positions = 0
            while seq_ids[word_start_char_positions] != (1 ):
                word_start_char_positions += 1

            word_end_char_positions = len(input_ids) - 1
            while seq_ids[word_end_char_positions] != (1 ):
                word_end_char_positions -= 1

            if not (offset[word_start_char_positions][0] <= start_char and offset[word_end_char_positions][1] >= end_char):
                start_pos.append(cls_index)
                end_pos.append(cls_index)
            else:
                while word_start_char_positions < len(offset) and offset[word_start_char_positions][0] <= start_char:
                    word_start_char_positions += 1
                start_pos.append(word_start_char_positions - 1)
                while offset[word_end_char_positions][1] >= end_char:
                    word_end_char_positions -= 1
                end_pos.append(word_end_char_positions + 1)
    tokenized_list["start_positions"] = start_pos
    tokenized_list["end_positions"] = end_pos
    return tokenized_list


def train_model(datasets,model_checkpoint = "bert-large-uncased-whole-word-masking-finetuned-squad"):    
    initial_path= os.path.join(sys.path[0],'bert_large_trained_initial')
    model_checkpoint=initial_path if os.path.isdir(initial_path) else model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train,evaluate,test = np.split(datasets.sample(frac=1, random_state=42), [int(.6*len(datasets)),int(.8*len(datasets))])
    #datasets = datasets.applymap(str)
    test.to_excel(os.path.join(sys.path[0],"qa_test.xlsx"))
    s
    #question_train=train['question'].values.tolist()
    #context_train=train['context'].values.tolist()
    train['answer']=train.answer.apply(lambda x: ast.literal_eval(str(x)))
    #answer_train=[ast.literal_eval(x) for x in train['answer'].values]
    #question_eval=test['question'].values.tolist()
    #context_eval=test['context'].values.tolist()
    test['answer']=test.answer.apply(lambda x: ast.literal_eval(str(x)))
    evaluate['answer']=evaluate.answer.apply(lambda x: ast.literal_eval(str(x)))

    #answer_test=[ast.literal_eval(x) for x in test['answer'].values]
    train_dataset=Dataset.from_pandas(train)
    
    eval_dataset=Dataset.from_pandas(evaluate)
    train_tokenized_dataset = train_dataset.map(add_tokenised_features, batched=True, remove_columns=train_dataset.column_names,fn_kwargs={'tokenizer':tokenizer})
    eval_tokenized_dataset = eval_dataset.map(add_tokenised_features, batched=True, remove_columns=eval_dataset.column_names,fn_kwargs={'tokenizer':tokenizer})

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    args = TrainingArguments(
        output_dir=os.path.join(sys.path[0],'bert_large_trained_initial'),
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=5,
        weight_decay=0.01,    
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=eval_tokenized_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        
    )
    trainer.train(resume_from_checkpoint = True)
    trainer.save_model(os.path.join(sys.path[0],'bert_large_trained_tqa_final'))
    
if __name__=="__main__":
   unloaded_json=download_zip()
   initial_json=json.load(unloaded_json)
   datasets=transform_json(initial_json)
   #datasets=pd.read_excel("/content/drive/MyDrive/qa.xlsx",index_col=0)
   train_model(datasets)     