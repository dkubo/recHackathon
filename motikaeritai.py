
# coding: utf-8

# In[66]:


# from gensim.models import word2vec
import numpy as np
import pickle
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from collections import defaultdict
from sklearn import tree
import pydotplus
from graphviz import Digraph
from sklearn.externals.six import StringIO


# **ダミー変数化**

def transDummy(idx, data, x):
    dum_posi = pd.DataFrame(pd.get_dummies(data[idx]))
    x = pd.concat([x, dum_posi], axis=1)
    return x


# **データ整形**

def shapeforAki(testhash, fname):
    with open(fname, "a") as f:
# #         既存カテゴリは、先頭から順に階層が下がっていく
        for (i, v) in testhash.items():
# #             f.write(str(url)+"\t"+str(v["title"])+"\t"+",".join(v["existing_category"])+"\t"+",".join(v["fromtitle_category"])+"\n")
            f.write(str(v["url"])+"\t"+str(v["title"])+"\t"+"\t".join(v["existing_category"])+"\n")


# **決定木グラフ描画**

def graph(clf, categories, fname):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,feature_names=categories)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(fname)


# **対話部分**

def genQustion(category):
    print('> 「' + category + '」に関連していますか? (はい / いいえ)' )
    # print '> 「' + category + '」に関連していますか? (はい / いいえ / ひとつ戻る)'

def getResponse():
    return input()

def printTargets(rest_targets, targets):
    print('----------------------')
    for idx in rest_targets:
        print(targets[idx])
    print('----------------------')
    
def dialogue(clf, categories, targets, leafidx):
    position = [0]
    while position[-1] not in leafidx:
        print(position)
        genQustion(categories[clf.tree_.feature[position[-1]]])
        response = getResponse()
        if response == 'はい':
            position.append(clf.tree_.children_right[position[-1]])
        elif response == 'いいえ':
            position.append(clf.tree_.children_left[position[-1]])
        # elif response == '一つ戻る':
        # 	position.pop()
        # else:
        # 	print '「はい」もしくは「いいえ」でお願いします。。！	'

        # 結果の表示
        if clf.tree_.n_node_samples[position[-1]] < 100:
            rest_targets = [i for i, v in enumerate(clf.tree_.value[position[-1]][0]) if v == 1]
            printTargets(rest_targets, targets)
    last_position = position[-1]
    if clf.tree_.n_node_samples[last_position] >= 100:
        rest_targets = [i for i, v in enumerate(clf.tree_.value[last_position][0]) if v == 1]
        printTargets(rest_targets, targets)

# **#### main ####**

# ** アキネーター **
# # 決定木の分類器をCARTアルゴリズムで作成
# clf = tree.DecisionTreeClassifier(max_depth=10, random_state=31, criterion='gini')
clf = tree.DecisionTreeClassifier(max_depth=10, random_state=31, min_samples_leaf=1, criterion='entropy')
clf = clf.fit(x, y)
# data.save('../data/clf.pickle', clf)
# clf = data.load("../data/clf.pickle")

# 決定木可視化
# graph(clf, x.columns, "../data/akinator_graph.pdf")

leafidx = [i for i, v in enumerate(clf.tree_.feature) if v == -2]	# リーフのインデックス取得
# # leafsamples = [clf.tree_.n_node_samples[idx] for idx in leafidx]

# # # インタラクション開始
dialogue(clf, x.columns, y, leafidx)

# # #モデルを用いて予測を実行する
# # print 'evaluate model ....'
# # #識別率を確認
# # result = sum(predicted == y) / len(y)