import os,sys
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
#from transformers import AutoTokenizer
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,pipeline
import pandas as pd
from utils.doc_query import sentence_splitter,tf_idf_similarity_check
############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    model_path=os.path.join(sys.path[0],'utils/bert_large_trained_tqa_final')
    model_path=model_path if os.path.exists(model_path) else 'tvsharish/bert-large-finetuned-tqa'
    #tvsharish/bert-large-finetuned-tqa is my model checkedin for easier access
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
    df=pd.read_excel(os.path.join(sys.path[0],"utils/qa_test.xlsx"),engine='openpyxl',index_col=0)
    context=list(set(df["context"].values.tolist()))
    sentence_list=sentence_splitter(context)
    for text in request.text:
        # TODO Add code here
        _score,context=tf_idf_similarity_check(text,sentence_list)
        result = nlp(question = text, context=context)
        response = result.get('answer','')
        output.append(response)

    return SimpleText(dict(text=output))
