- **MLMN:**
  - main.py: train MLMN
  - parameters.py: set relevant configuration parameters
  - preprocess.py: preprocess data
  - models.py: the specific architecture of MLMN
  - baseline_models.py: reproduction of baseline models
  - train.py: train tools
  - utils.py: other basic tools
- **The classifier for predicting the parsed information of law articles**:
  - article_condition_classifier.py: train the classifier
  - article_rules.py: the rules of whether text is premise or conclusion
  - predict_pre_con.py: input articles and output the parsed information
- **Decision predictor:**
  - decision_predictor.py: train the model for predicting judicial decision

- **data:**

  - law common words.txt & law_dataset.json: corpus used to train the parsed information classifier

  - labeled_corpus.txt: the corpora only contain fine-grained fact-article pairs

  - hurt_corpus.json & traffic_corpus.json:

    ```json
    [
        {
            'file_name': 'the name of the original file',
            'paragraph': 'the extracted complete fact paragraph',
            'sentences': {
                'fact1': [ids of labelled articles],
                ……,
                'factn': [ids of labelled articles]
            },
            'articles':[ids of articles cited by original document],
            'decision': the number of months of fixed-term imprisonment
        }
    ]
    ```