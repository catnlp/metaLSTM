# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/18 12:18
'''
import os

def addContent(target, file, tag):
    print(file)
    beginTag = '<' + tag + '>'
    endTag = '</' + tag + '>'
    with open(file) as src:
        lines = src.read()
        lines = lines.replace('\n\n', '\n'+endTag+'\tS-'+tag+'\n\n'+beginTag+'\tS-'+tag+'\n')
        lines = beginTag + '\tS-'+tag+'\n' + lines + endTag + '\tS-'+tag+'\n\n'
        target.write(lines)

def make_jointCorpus(dataset, name, dirList):
    print('---make joint corpus---')
    if dataset[-1] == '/':
        dataset = dataset[0: -1]

    joint_dir = dataset + '/' + name
    if not os.path.exists(joint_dir):
        os.makedirs(joint_dir)

    trainF = joint_dir + '/train.tsv'
    develF = joint_dir + '/devel.tsv'
    testF = joint_dir + '/test.tsv'

    with open(trainF, 'w') as trainF, open(develF, 'w') as develF, open(testF, 'w') as testF:
        for dir in dirList:
            tag = dir
            dir = dataset + '/' + dir + '-IOBES'
            addContent(trainF, dir+'/train.tsv', tag)
            addContent(develF, dir+'/devel.tsv', tag)
            addContent(testF, dir+'/test.tsv', tag)

if __name__ == "__main__":
    dataset = '../../data/group/species'
    name = 'joint-species'
    dirList = ['BioNLP11ID-species', 'BioNLP13CG-species', 'CRAFT-species', 'linnaeus']
    make_jointCorpus(dataset, name, dirList)