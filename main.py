from models import *
from baseline_models import *
from train import process_data, train_model, train_model_with_element
from parameters import parse
from utils import load_txt
import numpy as np
import random

if __name__ == '__main__':
    np.random.seed(2233)
    torch.manual_seed(2233)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2233)

    random.seed(2233)

    args = parse()

    stop_words = [line.strip() for line in load_txt(args.stopwords_path)]
    vocab = load_pkl(args.vocab_path)

    train_data = load_txt(args.train_corpus_for_recommendation)
    train_data = process_data(train_data, stop_words, vocab)

    valid_data = load_txt(args.valid_corpus_for_recommendation)
    valid_data = process_data(valid_data, stop_words, vocab)

    test_data = load_txt(args.test_corpus_for_recommendation)
    test_data = process_data(test_data, stop_words, vocab)

    if 'WithElement' in args.model_name:
        model = globals()[args.model_name](args)
        train_model_with_element(model, args, train_data, valid_data, test_data)
    else:
        model = globals()[args.model_name](args)
        train_model(model, args, train_data, valid_data, test_data)
