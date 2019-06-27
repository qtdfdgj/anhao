import networkx as nx
import pandas as pd
import operator
import numpy as np


def k_shell(graph):
    importance_dict = {}
    level = 1
    while len(graph.degree):
        importance_dict[level] = []
        while True:
            level_node_list = []
            for item in graph.degree:
                if item[1] <= level:
                    level_node_list.append(item[0])
            graph.remove_nodes_from(level_node_list)
            importance_dict[level].extend(level_node_list)
            if not len(graph.degree):
                return importance_dict
            if min(graph.degree, key=lambda x: x[1])[1] > level:
                break
        level = min(graph.degree, key=lambda x: x[1])[1]
    return importance_dict


def changeForm(result):#将k-shell的键值互换
    temp = {}
    for key in result.keys():
        a = result.get(key)
        if len(a) != 0:
            for i in a:
                temp.update({i: key})
    return temp


def extentEdge(wEdge,wFactor):#扩展边的权值集合，比如节点1的(1,2)边，在计算节点2的时候没有(2,1)边，所以加上(2,1)边
    temp = {}
    temp2={}
    for key in wEdge.keys():
        temp.update({(key[1],key[0]):wEdge.get(key)})
        temp2.update({(key[1], key[0]): wFactor.get(key)})
    temp.update(wEdge)
    temp2.update(wFactor)
    return temp,temp2


def getEdgeWeight(graph):  # 获取边的权重
    edge = list(graph.edges)
    degreedict = dict(list(graph.degree))  # 获取每个节点的度数
    w_edge = {}
    for i in edge:  # 求出每个边对应的权值，以字典的形式存入w_edge
        num = float(degreedict.get(i[0]) + degreedict.get(i[1]))
        temp = {i: num}
        w_edge.update(temp)
    return w_edge


def getNodeFactor(graph):  # 获取边的影响系数
    sorted_nodes = sorted(graph.nodes())
    nodes_neighbours = {}
    for i in sorted_nodes:
        nodes_neighbours.update({i: list(graph.neighbors(i))})
    edge = graph.edges
    factor = {}
    for i in edge:
        join = set(nodes_neighbours.get(i[0])).intersection(set(nodes_neighbours.get(i[1])))  # 求交集
        union = set(nodes_neighbours.get(i[0])).union(set(nodes_neighbours.get(i[1])))  # 求并集
        union.remove(i[0])  # 移除左端点
        union.remove(i[1])  # 移除右端点
        factor.update({i: (len(join) + 1) / (len(union) + 1)})  # 求
    return factor, nodes_neighbours


def editNode(neighbors,ks):#删除ks小于本身的邻居节点
    neighborsFC={}
    for i in neighbors.keys():
        temp=neighbors.get(i)
        temp2=temp.copy()
        for j in temp:
            if(ks.get(j)<ks.get(i)):
                temp2.remove(j)
        neighborsFC.update({i:temp2})
    return neighborsFC


def getWk(node, neighbours, ks, w, I):  # ks是k-shell值，w是边的权重，I节点的系数，字典类型
    wk = {}
    neighbours=editNode(neighbours,ks)
    for i in node:
        temp = neighbours.get(i)
        numw = 0.0
        num2 = 0.0
        for j in temp:
            a = (i, j)
            numw += w.get(a)
            num2 += float(I.get(a)) * float(ks.get(j))
        wk.update({i: round(numw + num2,3)})
    return wk


def getMks(dictwk):
    sorted_x=list(sorted(dictwk.items(),key=operator.itemgetter(1)))
    index=1
    mks={}
    for i in range(len(sorted_x)):
        if(i>0):
            if(sorted_x[i][1]!=sorted_x[i-1][1]):
                index+=1
        mks.update({sorted_x[i][0]:index})
    return mks

def getDegree(graph):
    dictgre=dict(graph.degree)
    sorted_x = dict(sorted(dictgre.items(), key=operator.itemgetter(0)))
    degree=list(sorted_x.values())
    return degree
def getOrder(dictitem):
    sorted_x = dict(sorted(dictitem.items(), key=operator.itemgetter(0)))
    order=list(sorted_x.values())
    return order

def saveToCsv(D,mks,wk,k_shell):
    wk_list = getOrder(wk)
    mks_list = getOrder(mks)
    k_shell_list=getOrder(k_shell)
    save = pd.DataFrame({'degree': D, 'mks': mks_list, 'wk': wk_list,'k_shell':k_shell_list})
    save.index = np.arange(1, len(save) + 1)
    save.to_csv("result.csv")#存入result文件中


if __name__ == '__main__':
    data = pd.read_csv('test.csv', header=0)
    graph = nx.Graph()
    data_len = len(data)
    for i in range(data_len):
        graph.add_edge(data.ix[i]['Source'], data.ix[i]['Target'])
    D=getDegree(graph)
    node1 = tuple(graph.nodes())#获取节点
    edgeWeight = getEdgeWeight(graph)#获取边的权重
    eFactor, neighbors = getNodeFactor(graph)#获取边的影响系数和节点的邻节点
    k_shellResult = k_shell(graph)#获取k-shell值
    k_shellResult2 = changeForm(k_shellResult)#k-shell结果的键值反转
    extendEdgeWeight,extentEFactor=extentEdge(edgeWeight,eFactor)#扩展边的权重和影响因素系数
    wk = getWk(node1, neighbors,  k_shellResult2,extendEdgeWeight,extentEFactor)#获取wk的值
    print("wk",wk)
    mks=getMks(wk)#获取mks
    print("MKS",mks)
    saveToCsv(D,wk,mks,k_shellResult2)#存入文件





































    '''
    画图
    pos=nx.random_layout(graph)
    nx.draw(graph,pos,with_labels=True,node_color="white",edge_color="red",node_size=400,alpha=0.5)
    pylab.show()
   
    '''
