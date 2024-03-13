import os
import fileinput
from doctor_offline.config import NEO4J_CONFIG
from py2neo import Graph, Node, Relationship

graph = Graph(**NEO4J_CONFIG)


def load_data(path):
    """
    path: 疾病csv文件路径
    功能：将csv文件数据读入缓存
    return：dic：疾病名：[症状1，症状2...]
    """
    disease_csv_list = os.listdir(path)
    disease_list = list(map(lambda x: x.split('.')[0], disease_csv_list))
    symptom_list = []
    for disease_csv in disease_csv_list:
        symptom = list(map(lambda x: x.strip(), fileinput.FileInput(os.path.join(path, disease_csv), encoding="utf-8")))
        symptom = list(filter(lambda x: 0 < len(x) < 100, symptom))
        symptom_list.append(symptom)
    return dict(zip(disease_list, symptom_list))


def write(path):
    """
    功能: 写入neo4j
    :param path: 疾病csv文件路径
    """
    disease_symptom_dict = load_data(path)
    for key, value in disease_symptom_dict.items():
        node1 = Node("Disease", name="key")
        graph.create(node1)
        for v in value:
            print("正在写入:" + key + v)
            node2 = Node("Symptom", name=v)
            graph.create(node2)
            relation = Relationship(node1, "dis_to_sym", node2)
            graph.create(relation)
    graph.run("CREATE INDEX FOR (n:Disease) ON (n.name)")
    graph.run("CREATE INDEX FOR (n:Symptom) ON (n.name)")


if __name__ == "__main__":
    path = './data/reviewed'
    write(path)
