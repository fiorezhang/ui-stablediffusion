# -*- coding: UTF-8 -*-

import requests

def ifWithChinese(string):
    for char in string:
        if '\u4e00' <= char <= '\u9fa5':
            return True
    return False

def translateYouDao(con):
    if ifWithChinese(con):
        try:
            data = {'doctype': 'json',
                    type: 'ZH_CN2EN',
                    'i': con}
            r = requests.get("https://fanyi.youdao.com/translate", params=data)
            res_json = r.json()
            res_d = res_json['translateResult'][0]
            tgt = []
            for i in range(len(res_d)):
                tgt.append(res_d[i]['tgt'])
            return ''.join(tgt)
        except Exception as e:
            print('Translate failed: ', e)
            return ""
    else:
        return con

if __name__ == '__main__':
    con = '戴黄帽子的小女孩看星星'
    res = translateYouDao(con)
    print('翻译结果：\n', res)