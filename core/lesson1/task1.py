"""
本文实现城市路径规划，虽数据源不适用，但功能逻辑是OK的
"""
import re
import math
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

# 如果图中汉字无法显示，请参照如下配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']


def get_location():
    """
    获取城市经纬度（这样的数据，在网络上很容易获得，主要为了方便说明主要原理，不再纠结数据源出处）
    :return:
    """
    coordination_source = """
    {name:'兰州', geoCoord:[103.73, 36.03]},
    {name:'嘉峪关', geoCoord:[98.17, 39.47]},
    {name:'西宁', geoCoord:[101.74, 36.56]},
    {name:'成都', geoCoord:[104.06, 30.67]},
    {name:'石家庄', geoCoord:[114.48, 38.03]},
    {name:'拉萨', geoCoord:[102.73, 25.04]},
    {name:'贵阳', geoCoord:[106.71, 26.57]},
    {name:'武汉', geoCoord:[114.31, 30.52]},
    {name:'郑州', geoCoord:[113.65, 34.76]},
    {name:'济南', geoCoord:[117, 36.65]},
    {name:'南京', geoCoord:[118.78, 32.04]},
    {name:'合肥', geoCoord:[117.27, 31.86]},
    {name:'杭州', geoCoord:[120.19, 30.26]},
    {name:'南昌', geoCoord:[115.89, 28.68]},
    {name:'福州', geoCoord:[119.3, 26.08]},
    {name:'广州', geoCoord:[113.23, 23.16]},
    {name:'长沙', geoCoord:[113, 28.21]},
    {name:'海口', geoCoord:[110.35, 20.02]},
    {name:'沈阳', geoCoord:[123.38, 41.8]},
    {name:'长春', geoCoord:[125.35, 43.88]},
    {name:'哈尔滨', geoCoord:[126.63, 45.75]},
    {name:'太原', geoCoord:[112.53, 37.87]},
    {name:'西安', geoCoord:[108.95, 34.27]},
    {name:'台湾', geoCoord:[121.30, 25.03]},
    {name:'北京', geoCoord:[116.46, 39.92]},
    {name:'上海', geoCoord:[121.48, 31.22]},
    {name:'重庆', geoCoord:[106.54, 29.59]},
    {name:'天津', geoCoord:[117.2, 39.13]},
    {name:'呼和浩特', geoCoord:[111.65, 40.82]},
    {name:'南宁', geoCoord:[108.33, 22.84]},
    {name:'西藏', geoCoord:[91.11, 29.97]},
    {name:'银川', geoCoord:[106.27, 38.47]},
    {name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
    {name:'香港', geoCoord:[114.17, 22.28]},
    {name:'澳门', geoCoord:[113.54, 22.19]}
    """

    city = {}
    for item in coordination_source.split('\n'):
        city_name = re.findall(r'name:\'(.*)?\'', item)
        location = re.findall(r'geoCoord:\[(\d+\.\d+).*\s(\d+\.\d+)\]', item)
        if len(city_name) > 0 and len(location) > 0:
            city[city_name[0]] = tuple(map(float, location[0]))
    return city


def geo_distance(origin, destination):
    """
    计算球面两点直线距离

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(geo_distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


def get_city_distance(city, city1, city2):
    """
    求两城市间距离
    :param city:
    :param city1:
    :param city2:
    :return:
    """
    return geo_distance(city[city1], city[city2])


def get_connect(city):
    """
    连接各城市节点
    :param city:
    :return:
    """
    city_connect = defaultdict(list)
    for key in city.keys():
        for add in city.keys():
            if add == key:
                continue
            if get_city_distance(city, key, add) < 1500:
                city_connect[key].append(add)
    return city_connect


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


def get_distance_by_path(path):
    """
    求城市list的距离
    :param path: ['兰州', '武汉', '上海']
    :return: 距离
    """
    distance = 0
    for index, item in enumerate(path):
        if index != 0:
            distance += get_city_distance(get_location(), path[index - 1], item)
    return distance


def sort_by_distance(paths):
    """
    对所有path路径正序排序
    :param paths: 多个路径[['兰州', '石家庄', '上海'], ['兰州', '武汉', '上海']]
    :return: 排序后的list
    """
    return sorted(paths, key=get_distance_by_path)


def init_data1():
    '''
    初始化一个简单的图数据
    :return: 返回字典，其key为某个人姓名，value为其认识的所有人
    '''
    init = {
        '小张': ['小刘', '小王', '小红'],
        '小王': ['六六', '娇娇', '小曲'],
        '娇娇': ['宝宝', '花花', '喵喵'],
        '六六': ['小罗', '奥巴马']
    }
    social_network = defaultdict(list)
    for key in init.keys():
        social_network[key] = init[key]
    return social_network


def init_data2():
    """
    初始化城市图数据
    :return:
    """
    city = get_location()
    city_connect = get_connect(city)
    print(city_connect)
    return city_connect


city_connect = init_data2()  # 初始化城市图数据（数据获取方式与数据本身适用性不高，但足以说明思想要点）
result = search(city_connect, "兰州", "上海")  # 搜索所有路径
print(result)
sorted = sort_by_distance(result)  # 对路径排序（求最短路径）
for item in sorted:
    print(get_distance_by_path(item), item)

# 打印结果(画图)，方便预览
city_connect_graph = nx.Graph(city_connect)
nx.draw(city_connect_graph, get_location(), with_labels=True)
all_path = []
for index1, item1 in enumerate(sorted):
    if index1 < 1:
        for index2, item2 in enumerate(item1):
            if index2 != 0:
                all_path.append((item1[index2-1], item2))
print(all_path)
nx.draw_networkx_edges(city_connect_graph, get_location(), all_path, width=2, edge_color='r')
plt.show()
