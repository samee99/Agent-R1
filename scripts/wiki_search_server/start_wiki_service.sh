#!/bin/bash

# 设置环境变量
export INDEX_PATH="../../data/corpus/wiki/wiki_index_IVF4096_PQ64.bin"
export CORPUS_PATH="../../data/corpus/wiki/wiki_corpus.jsonl"
export WIKI_SEARCH_API_URL="http://localhost:8000"

# 禁用代理
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# 安装依赖
# pip install -r requirements.txt

# 启动API服务
echo "启动Wiki搜索API服务..."
python search_api.py 