# 如何利用python调用另一台只有互联网连接的电脑的接口
# 本机的ip地址是 183.213.144.202
# 本机的端口是 8080
# 本机的接口是 /test
#
# 另一台电脑的ip地址是
#
# 另一台电脑的端口是
# 另一台电脑的接口是 /test
#
# 本机的代码如下：
# import requests
# import json
# import time
#
# url = 'http://
# port = 8080
# path = '/test'
# url = url + ':' + str(port) + path
# print(url)
# data = {'name': 'test'}
# headers = {'Content-Type': 'application/json'}
# while True:
#     try:
#         r = requests.post(url, data=json.dumps(data), headers=headers)
#         print(r.text)
#         time.sleep(1)
#     except Exception as e:
#         print(e)
#         time.sleep(1)
#         continue
#
# 另一台电脑的代码如下：
# import json
# from flask import Flask, request
# from flask_cors import CORS
# import time
#
# app = Flask(__name__)
# CORS(app, supports_credentials=True)
#
# @app.route('/test', methods=['POST'])
# def test():
#     data = request.get_data()
#     data = json.loads(data)
#     print(data)
#     return 'ok'
#
# if __name__ == '__main__':
#     app.run(host='
#             port=8080,
#             debug=True)
#
# 本机的代码可以正常运行，但是另一台电脑的代码一直报错：

# 已知服务器IP地址，端口号，接口，如何利用python调用服务器的接口
# 服务器的IP地址是
# 服务器的端口号是
# 服务器的接口是
#

# 本机的代码如下：
# import requests
# import json
# import time
#
# url = 'http://
# port = 8080
# path = '/test'
# url = url + ':' + str(port) + path
# print(url)
# data = {'name': 'test'}
# headers = {'Content-Type': 'application/json'}
# while True:
#     try:
#         r = requests.post(url, data=json.dumps(data), headers=headers)
#         print(r.text)
#         time.sleep(1)
#     except Exception as e:
#         print(e)
#         time.sleep(1)
#         continue
#
# 服务器的代码如下：


