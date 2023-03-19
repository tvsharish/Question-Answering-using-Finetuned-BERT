import json,sys,os
import pandas as pd
from rapidfuzz import fuzz
import nltk.data
from rapidfuzz import utils
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')

def transform_json(initial_json):
    df=pd.DataFrame(columns=["context","question","answer"])
    try:
       for index,element in enumerate(initial_json):
        if all(["questions" in element.keys(),"topics" in element.keys()]):
            if len(element["questions"].get("nonDiagramQuestions",[]))>0:
               topic=""
               for key,value in element["topics"].items():
                   topic+=value["content"]["text"]
               for key,value in element["questions"].get("nonDiagramQuestions",[]).items():
                   ques_ans_json=element["questions"]["nonDiagramQuestions"][key]
                   question=ques_ans_json.get("beingAsked","").get("processedText","")
                   answer_key=ques_ans_json.get("correctAnswer","").get("processedText","")
                   answer=ques_ans_json.get("answerChoices",{}).get(answer_key,{}).get("processedText","")
                   
                   answer_type=ques_ans_json.get("questionSubType","")
                   if answer_type=="True or False":
                      continue
                   if "of the above" in answer or "of these" in answer:
                      if "all of the above" in answer or "all of these" in answer:
                        for key1,value1 in ques_ans_json.get("answerChoices","").items():
                            if key1!=answer_key:
                                answer=value1.get("processedText","")
                                score,similar_text=fuzzy_similarity(topic,answer,score=60)
                                answer_index=find_answer_index(topic,similar_text,answer)
                                answer_block={'answer_start': [answer_index], 'text': [answer]}
                                break
                      else:
                        continue          
                   else:
                      score,similar_text=fuzzy_similarity(topic,answer,score=60)
                      answer_index=find_answer_index(topic,similar_text,answer)
                      answer_block={'answer_start': [answer_index], 'text': [answer]}                      
                   if answer_index==-1:
                      continue
                   df=df.append({"question":question,"answer":answer_block,"context":topic},ignore_index = True)
    except Exception :
       import traceback
       print(traceback.format_exc())                
    df.to_excel(os.path.join(sys.path[0],"qa.xlsx"))
    return df                  


def find_answer_index(topic,similar_text,answer):
    similar_text=utils.default_process(similar_text)
    answer=utils.default_process(answer)
    similar_text_list=similar_text.split()
    common_word=sorted(set(similar_text_list)&set(answer.split()),key=similar_text_list.index)
    common_word=' '.join(common_word)
    try:
       if len(common_word)==0:
          common_word=[x for x in similar_text_list if answer in x][0]
    
       similar_text_index=topic.lower().index(common_word)
       return similar_text_index
    except Exception :
       return -1                     

def fuzzy_similarity(text,phrase,score=80):
    list_of_scores=[]
    for i in tokenizer.tokenize(text):
        if len(i.strip())==0:
           continue
        r=fuzz.partial_ratio(i,phrase)
        if r>score:
           list_of_scores.append([r,i])
    return sorted(list_of_scores,key=lambda i:i[0],reverse=True)[0] if len(list_of_scores)!=0 else [0,'']                              


if __name__=="__main__":
   initial_json=json.load(open("/home/htd/Downloads/a.json","r"))
   transform_json(initial_json) 

