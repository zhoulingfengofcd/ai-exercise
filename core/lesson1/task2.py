"""
北京地铁搜索项目
"""
import requests
import re
import numpy as np
import json
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

# 如果图中汉字无法显示，请参照如下配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']


def get_lines_stations_info(text):
    # 请在这里写上你的代码
    jsondata = json.loads(text)
    lines_list = jsondata['l']


    # 遍历text格式数据，组成地点数据结构
    # 所有线路信息的dict：key：线路名称；value：站点名称list
    lines_info = {}

    # 所有站点信息的dict：key：站点名称；value：站点坐标(x,y)
    stations_info = {}

    #for i in range(len(lines_list)):
        # 你可能需要思考的几个问题，获取「地铁线路名称，站点信息list，站名，坐标(x,y)，数据加入站点的信息dict，将数据加入地铁线路dict」
        #pass
    for i in range(len(lines_list)):
        line_name = lines_list[i]['ln']  # 线路名称
        line_code = lines_list[i]['ls']  # 线路编号
        all_site = lines_list[i]['st']  # 所有站点
        site_list = []
        for j in range(len(all_site)):
            site_name = all_site[j]['n']  # 站点名称
            position = all_site[j]['sl']  # 定位
            site_all_line_code = all_site[j]['r']  # 站点连接的所有线路
            site_list.append(site_name)
            stations_info[site_name] = tuple(map(float, position.split(',')))
        lines_info[line_name] = site_list
    return lines_info, stations_info


# 根据线路信息，建立站点邻接表dict
def get_neighbor_info(lines_info):
    neighbor_info = defaultdict(list)
    for line_name in lines_info.keys():
        for index2 in range(len(lines_info[line_name])-1):
            site1 = lines_info[line_name][index2]
            site2 = lines_info[line_name][index2+1]
            neighbor_info[site1].append(site2)
            neighbor_info[site2].append(site1)
            # if index2 == 0:
            #     neighbor_info[site].append(lines_info[line_name][index2+1])
            # elif index2 < len(lines_info[line_name])-2:
            #     neighbor_info[site].append(lines_info[line_name][index2 + 1])
            #     neighbor_info[site].append(lines_info[line_name][index2 - 1])
            # else:
            #     neighbor_info[site].append(lines_info[line_name][index2 - 1])
    return neighbor_info


def search(graph, start, end):
    '''
    从图数据graph中，搜索start->end的所有路径
    :param graph: 图数据，格式如下：
    {
        '小张': ['小刘', '小王', '小红'],
        '小王': ['六六', '娇娇', '小曲'],
        '娇娇': ['宝宝', '花花', '喵喵'],
        '六六': ['小罗', '奥巴马']
    }
    :param start: 开始节点
    :param end: 结束节点
    :return: 所有路径列表
    '''

    # 检查输入站点名称
    if not graph.get(start):
        print('起始站点“%s”不存在。请正确输入！' % start)
        return None
    if not graph.get(end):
        print('目的站点“%s”不存在。请正确输入！' % end)
        return None

    check = [[start]]  # 搜索路径
    visited = set()  # 已访问过的节点
    finished = []
    while check:
        # 1、取一条未完成的搜索路径
        # 队列（广度优先搜索）
        path = check.pop(0)  # 取队列头
        # 栈（深度优先搜索）
        # paths = check.pop(-1)  # 取栈顶

        # 2、遍历该路径的下一层
        node = path[-1]
        # 如果节点已访问，则抛弃（本条路径），防止死循环
        if node in visited:
            continue
        # 遍历下一层
        for item in graph[node]:
            new_path = path + [item]

            if item == end:
                finished.append(new_path)  # 找到终点，则该路径结束搜索
            else:
                check.append(new_path)  # 否则将路径添加到搜索路径中
        visited.add(node)
    return finished


def get_distance(stations_info, str1, str2):
    x1, y1 = stations_info.get(str1)
    x2, y2 = stations_info.get(str2)
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5


def get_distance_by_path(stations_info):
    """
    求城市list的距离
    :param path: ['兰州', '武汉', '上海']
    :return: 距离
    """
    def function(path):
        distance = 0
        for index, item in enumerate(path):
            if index != 0:
                distance += get_distance(stations_info, path[index - 1], item)
        return distance
    return function


def sort_by_distance(stations_info, paths):
    """
    对所有path路径正序排序
    :param paths: 多个路径[['兰州', '石家庄', '上海'], ['兰州', '武汉', '上海']]
    :return: 排序后的list
    """
    return sorted(paths, key=get_distance_by_path(stations_info))


# 1、数据准备
r = requests.get('http://map.amap.com/service/subway?_1587969944497&srhdata=1100_drw_beijing.json')
lines_info, stations_info = get_lines_stations_info(r.text)
print(lines_info, stations_info)
#connect_graph = nx.Graph(lines_info)
#nx.draw(connect_graph, stations_info, with_labels=True)
neighbor_info = get_neighbor_info(lines_info)
print(neighbor_info)
connect_graph = nx.Graph(neighbor_info)
#nx.draw(connect_graph, stations_info, with_labels=True, node_size=6, font_size=8)
#plt.show()

# 2、站点搜索
search_paths = search(neighbor_info, '宣武门', '双合')
print("搜索结果", search_paths)
sorted = sort_by_distance(stations_info, search_paths)
for item in sorted:
    print(get_distance_by_path(stations_info)(item), item)

# 3、打印结果(画图)，方便预览
connect_graph = nx.Graph(neighbor_info)
nx.draw(connect_graph, stations_info, with_labels=True, node_size=6, font_size=8)
all_path = []
for index1, item1 in enumerate(sorted):
    if index1 < 1:
        for index2, item2 in enumerate(item1):
            if index2 != 0:
                all_path.append((item1[index2-1], item2))
print(all_path)
nx.draw_networkx_edges(connect_graph, stations_info, all_path, width=2, edge_color='r')
plt.show()