import time
import requests
from bs4 import BeautifulSoup
import re
import csv

BASE_URL = 'http://www.tianqihoubao.com/aqi/'


def get_city_coding():
    """
    获取所有城市与编号字典
    :return: { ‘杭州’ : 'hangzhou', '广州' : 'guangzhou'}
    """
    response = requests.get(BASE_URL)

    soup = BeautifulSoup(response.text.encode(response.encoding), features="html.parser")
    all_city = re.findall(r'/aqi/(\w*).html">(.*?)</a>', str(soup.select(".citychk")[0]))

    city_coding = {}
    for item in all_city:
        city_coding[item[1].strip()] = item[0].strip()
    return city_coding


def build_url(city, year=None, month=None):
    """
    构建对应城市的请求url
    :param city:
    :param year:
    :param month:
    :return: 格式 http://www.tianqihoubao.com/aqi/chongqing-201907.html or http://www.tianqihoubao.com/aqi/chongqing.html
    """
    if year is not None and month is not None:
        return (BASE_URL + '{}-{}{}.html').format(city, year,
                                                  '0'+str(month) if month < 10 else month)
    else:
        return (BASE_URL + '{}.html').format(city)


def get_http_content(url):
    """
    根据url获取空气质量soup数据
    :param url:
    :return:
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text.encode(response.encoding), features="html.parser")
    return soup.table.contents


def parse(contents, city):
    """
    解析soup数据，返回空气质量列表数据
    :param contents:
    :param city:
    :return:
    """
    data = []
    count = 0
    for item in contents:
        if hasattr(item, 'text'):
            data.append((['城市']+item.text.split()) if count == 0 else [city]+item.text.split())
            count += 1
    return data


def save(data, filename, mode='w'):
    """
    将list数据保存csv文件
    :param data:
    :param filename:
    :param mode:
    :return:
    """
    csv_file = open(r'./'+filename+'.csv', mode=mode, encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    for item in data:
        csv_writer.writerow(item)


def get_air_quality(city=None, year=None, month=None):
    """
    获取对应城市空气质量
    :param city: 可为空，默认获取所有城市
    :param year: 可为空，year与month任意一项为空，默认获取最新数据
    :param month: 可为空，year与month任意一项为空，默认获取最新数据
    :return:
    """
    city_coding = get_city_coding()
    if city is None:
        for index, request_city in enumerate(city_coding.keys()):
            if index < 10:  # 爬全部城市，比较慢，测试时，限制下
                url = build_url(city_coding[request_city], year, month)
                contents = get_http_content(url)
                data = parse(contents, request_city)
                if index != 0:
                    data.pop(0)
                save(data, 'all_city', 'a')
                time.sleep(1)
    else:
        request_city_coding = city_coding[city]
        url = build_url(request_city_coding, year, month)
        contents = get_http_content(url)
        data = parse(contents, city)
        save(data, request_city_coding)


"""
获取城市空气质量，保存为csv文件
"""
get_air_quality('成都', year=2019, month=1)
