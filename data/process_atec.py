import codecs,pandas as pd
from data.data_utils import read_cut_file,saveDict
import matplotlib.pyplot as plt
import re,jieba
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False


def combine_train_files(dest_path="atec/atec_nlp_sim_train_all.csv",
                        src_path_list=("atec/atec_nlp_sim_train.csv","atec/atec_nlp_sim_train_add.csv")):
    final_file=codecs.open(dest_path,'w',encoding="utf-8")
    line_id=0
    for id,src_path in enumerate(src_path_list):
        if id==0:
            with codecs.open(src_path,'r',encoding="utf-8") as f:
                line=f.readline()
                while line:
                    final_file.write(line)
                    line_id+=1
                    line = f.readline()
        else:
            print(line_id) # 39346
            with codecs.open(src_path,'r',encoding="utf-8") as f:
                line=f.readline()
                now_id=None
                while line:
                    line=line.strip().split('\t')
                    now_id=str(int(line[0])+line_id)
                    final_file.write("\t".join([now_id]+line[1:])+"\n")
                    line = f.readline()
                line_id=int(now_id)
    final_file.close()
    with codecs.open(dest_path,'r',"utf-8") as f:
        data=f.readlines()
        print(len(data)) # 102477
        print(data[-1].split("\t"))


def get_t2s_dict(t_file="atec/atec_nlp_sim_train_all.csv",
                 s_file="atec/atec_nlp_sim_train_all.simp.csv",
                 save_path="atec/t2s_dict.json"):
    t2s={}
    with codecs.open(t_file,'r',"utf-8") as f:
        raw_data_t=f.readlines()
    with codecs.open(s_file,'r',"utf-8") as f:
        raw_data_s=f.readlines()
    assert len(raw_data_t)==len(raw_data_s)
    for l1,l2 in zip(raw_data_t,raw_data_s):
        l1=''.join(l1.split('\t')[1:3])
        l2=''.join(l2.split('\t')[1:3])
        assert len(l1)==len(l2)
        for t,s in zip(l1,l2):
            if t!=s:
                if t not in t2s.keys():
                    print("%s -> %s" % (t, s))
                    t2s[t]=s
                else:
                    assert s==t2s[t]
    saveDict(t2s,save_path)


def show_not_ch(file_path="atec/atec_nlp_sim_train_all.simp.csv",only_en=False):
    not_ch = {}
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        print('open a file.')
        lineNum = 1
        line = f.readline()
        while line:
            print('---processing ', lineNum, ' article---')
            line = line.strip().split('\t')
            if only_en:
                line1 = re.findall('[a-zA-Z]+', line[1])
                line2 = re.findall('[a-zA-Z]+', line[2])
            else:
                line1 = re.findall('[^\u4e00-\u9fa5]+', line[1])
                line2 = re.findall('[^\u4e00-\u9fa5]+', line[2])
            for w in line1+line2:
                try:
                    not_ch[w]+=1
                except KeyError:
                    not_ch[w]=1
            lineNum+=1
            line = f.readline()
    print(sorted(not_ch.items(),key=lambda x:x[1],reverse=True))


def label_distribution(trainFile="atec/training.csv"):
    """
    分析训练数据中标签分布情况
    """
    labels=read_cut_file(file_path=trainFile,with_label=True)["label"]
    neg_count=labels.count(0)
    pos_count=labels.count(1)
    assert neg_count+pos_count==len(labels)
    counts=[neg_count,pos_count]
    labels=["不同义","同义"]
    fig=plt.figure(figsize=(9,9))
    # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.pie(counts, labels=labels, autopct='%1.2f%%')
    plt.title("标签分布", bbox={'facecolor': '0.6', 'pad': 5})
    plt.show()
    savePath=trainFile.split(".")[0]+"_ld.png"
    fig.savefig(savePath)
    plt.close()


def sentence_length_distribution(trainFile="atec/training.csv"):
    """
    分析训练数据中句子长度分布
    """
    raw_data = read_cut_file(file_path=trainFile, with_label=True)
    df=pd.DataFrame(raw_data)
    level=["w","c"]
    for l in level:
        s1="sent1"+l+"_len"
        print(df[s1].describe())
        s2="sent2"+l+"_len"
        print(df[s2].describe())
        df_=pd.DataFrame({s1:df[s1],s2:df[s2]})
        fig=plt.figure(figsize=(32,18))
        df_.boxplot()
        plt.legend()
        plt.show()
        fig.savefig(trainFile.replace(".csv","_sl_"+l+".png"))


def get_corpus(file_path="atec/atec_nlp_sim_train_all.simp.csv",corpus_path="atec/atec"):
    """
    根据给定数据生成字级和词级语料库
    """
    jieba.load_userdict("myDict.txt")
    target_char = codecs.open(corpus_path+"_char", 'w', encoding='utf-8')
    target_word = codecs.open(corpus_path + "_word", 'w', encoding='utf-8')
    en2ch={"huabei":"花呗","jiebei":"借呗","mayi":"蚂蚁","xiugai": "修改", "zhifu": "支付",
           "zhifubao":"支付宝","mobike": "摩拜","zhebi":"这笔","xinyong":"信用","neng":"能",
           "buneng":"不能","keyi":"可以","tongguo":"通过","changshi":"尝试","bunengyongle":"不能用了",
           "mobie": "摩拜","feichang":"非常","huankuan":"还款","huanqian":"还钱","jieqian":"借钱",
           "shouqian":"收钱","shoukuan":"收款"}
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        print('open a file.')
        lineNum = 1
        line = f.readline()
        while line:
            print('---processing ', lineNum, ' article---')
            for k,v in sorted(en2ch.items(),key=lambda x:len(x[0]),reverse=True):
                line = line.replace(k, v)
            line = line.strip().split('\t')
            # 保留中文、英文
            p = re.compile(u'[^\u4e00-\u9fa5a-z]')  # 中文的编码范围是：\u4e00到\u9fa5
            line1 = " ".join(p.split(line[1].lower())).strip()
            line2 = " ".join(p.split(line[2].lower())).strip()
            w1=[w.strip() for w in jieba.cut(line1) if len(w.strip())>0]
            w2=[w.strip() for w in jieba.cut(line2) if len(w.strip())>0]
            target_char.write(" ".join([_ for _ in "".join(w1)])+"\n")
            target_char.write(" ".join([_ for _ in "".join(w2)])+"\n")
            target_word.write(" ".join(w1)+"\n")
            target_word.write(" ".join(w2)+"\n")
            lineNum = lineNum + 1
            line = f.readline()
    print('well done.')
    target_char.close()
    target_word.close()


if __name__=="__main__":

    # combine_train_files()

    # get_t2s_dict()

    # show_not_ch()
    # show_not_ch(only_en=True)

    # from data.data_utils import participle
    # participle("atec/atec_nlp_sim_train_all.simp.csv","atec/training.csv",True,None)

    # label_distribution()
    # sentence_length_distribution()

    # from data.data_utils import split_train_val
    # split_train_val("atec/training.csv",10,19941229)

    get_corpus()
