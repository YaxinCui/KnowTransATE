from datasets import load_dataset


# dataset = load_dataset('conll2002', 'es')

# dataset = load_dataset('conll2003', download_mode="force_redownload")

dataset = load_dataset('amazon_reviews_multi', download_mode="force_redownload")

dataset = load_dataset('yelp_review_full', download_mode="force_redownload")

# 解决办法 https://github.com/huggingface/datasets/issues/726
