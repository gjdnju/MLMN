import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', dest='model_name', type=str, default='ThreeLayersWithElement')

    parser.add_argument('--corpus_path', dest='corpus_path', type=str, default='./data/hurt/labeled_corpus.txt')
    parser.add_argument('--vocab_path', dest='vocab_path', type=str, default='./data/vocab.pkl')
    parser.add_argument('--train_corpus_for_recommendation', dest='train_corpus_for_recommendation', type=str,
                        default='./data/hurt/split_data/train_data.txt')
    parser.add_argument('--valid_corpus_for_recommendation', dest='valid_corpus_for_recommendation', type=str,
                        default='./data/hurt/split_data/valid_data.txt')
    parser.add_argument('--test_corpus_for_recommendation', dest='test_corpus_for_recommendation', type=str,
                        default='./data/hurt/split_data/test_data.txt')
    parser.add_argument('--article_dict_path', dest='article_dict_path', type=str,
                        default='./data/hurt/article_dict_for_train.pkl')
    parser.add_argument('--article_dict_with_element_path', dest='article_dict_with_element_path', type=str,
                        default='./data/hurt/article_dict_for_train_with_element.pkl')
    parser.add_argument('--article_qhj_dict_path', dest='article_qhj_dict_path', type=str,
                        default='./data/hurt/article_qhj_dict.json')
    parser.add_argument('--embedding_matrix_path', dest='embedding_matrix_path', type=str,
                        default='./data/embedding_matrix.pkl')
    parser.add_argument('--stopwords_path', dest='stopwords_path', type=str, default='./data/stopwords.txt')
    parser.add_argument('--model_path', dest='model_path', type=str, default='./output/ThreeLayersWithElement.pt')
    parser.add_argument('--tensorboard_path', dest='tensorboard_path', type=str,
                        default='./output/ThreeLayersWithElement/tensorboard')
    parser.add_argument('--log_path', dest='log_path', type=str, default='./output/log.txt')

    # model parameter
    parser.add_argument('--fact_len', dest='fact_len', type=int, default=50)
    parser.add_argument('--article_len', dest='article_len', type=int, default=50)
    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128)
    parser.add_argument('--filters_num', dest='filters_num', type=int, default=128)
    parser.add_argument('--kernel_size_1', dest='kernel_size_1', type=int, default=2)
    parser.add_argument('--kernel_size_2', dest='kernel_size_2', type=int, default=4)
    parser.add_argument('--kernel_size_3', dest='kernel_size_3', type=int, default=8)
    parser.add_argument('--kernel_size_4', dest='kernel_size_4', type=int, default=10)
    parser.add_argument('--linear_output', dest='linear_output', type=int, default=64)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)

    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=64)

    parser.add_argument('--d_model', dest='d_model', type=int, default=128)
    parser.add_argument('--nhead', dest='nhead', type=int, default=2)
    parser.add_argument('--nhid', dest='nhid', type=int, default=128)

    # train parameter
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.01)

    parser.add_argument('--negtive_multiple', dest='negtive_multiple', type=int, default=5)

    args = parser.parse_args()

    return args
