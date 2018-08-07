import os,re,jieba,codecs,logging
from gensim.corpora import WikiCorpus


def xml2txt(in_path="wiki/zhwiki-latest-pages-articles.xml.bz2",
            out_path="wiki/wiki.zh.txt"):
    """
    将xml的wiki数据转换为txt格式
    """
    logger = logging.getLogger(os.path.basename(in_path))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running...")

    space = " "
    i = 0
    out = codecs.open(out_path, 'w',encoding='utf-8')
    # gensim里的维基百科处理类WikiCorpus
    wiki =WikiCorpus(in_path, lemmatize=False, dictionary=[])
    for text in wiki.get_texts():
        # 通过get_texts将维基里的每篇文章转换位1行text文本，并且去掉了标点符号等内容
        out.write(space.join(text) + "\n")
        i = i+1
        if (i % 10000 == 0):
            logger.info("Saving "+str(i)+" articles.")
    out.close()
    logger.info("Finished Saving "+str(i)+" articles.")


def get_corpus(file_path="wiki/wiki.zh.simp.txt",corpus_path="wiki/wiki"):
    """
    根据给定训练数据生成字级和词级语料库
    """
    jieba.load_userdict("myDict.txt")
    target_char = codecs.open(corpus_path+"_char", 'w', encoding='utf-8')
    target_word = codecs.open(corpus_path+"_word", 'w', encoding='utf-8')
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        print('open a file.')
        lineNum = 1
        line = f.readline()
        while line:
            print('---processing ', lineNum, ' article---')
            line = line.strip()
            # 保留中文
            p = re.compile(u'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
            line= " ".join(p.split(line.lower())).strip()
            target_char.write(" ".join([c for c in line])+"\n")
            line = [w for w in jieba.cut(line)]
            target_word.write(" ".join(line)+"\n")
            lineNum = lineNum + 1
            line = f.readline()
    print('well done.')
    target_char.close()
    target_word.close()


if __name__ == "__main__":

    # xml2txt()

    get_corpus()
