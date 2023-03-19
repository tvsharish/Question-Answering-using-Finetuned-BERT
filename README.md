# Question-Answering-using-Finetuned-BERT-openfabric-test-

I have finetuned [bert-large-uncased-whole-word-masking-finetuned-squad](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad) using non-Diagram questions-answers of [TQA Dataset](openfabric-test/utils/tqa_v1_train.json). The finetuned model is placed in [hugging-face-hub](https://huggingface.co/tvsharish/bert-large-finetuned-tqa).

I am using TF-IDF to find relevant context for the questions from the test dataset and context is passed to BERT finetuned model along with query to find answer. The dataset can be expanded to add relevant Science context for the queries. Presently, It have questions from Life Science, Earth Science and Physical Science textbooks of middle school science curricula. 