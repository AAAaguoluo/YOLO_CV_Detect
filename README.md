# YOLO_CV_Detect
这是一个学校视觉课程的课程设计项目，采用C/S结构，通过前后端分离，可以让在同一局域网内的设备访问该程序，基于YOLOv11模式实现了单张，批量，视频检测和实时摄像头识别功能。能实现对以下20种常见农业害虫的识别

This is a course design project for a school visual course, adopting a C/S structure. By separating the front-end and back-end, it enables devices within the same local area network to access the program. Based on the YOLOv11 mode, it realizes the functions of single image, batch, video detection and real-time camera recognition. It can identify the following 20 common agricultural pests

| 序号 | 名称       |
| ---- | ---------- |
| 0    | 稻纵卷叶螟 |
| 1    | 稻叶夜蛾   |
| 2    | 稻秆蝇     |
| 3    | 二化螟     |
| 4    | 三化螟     |
| 5    | 稻瘿蚊     |
| 6    | 稻秆潜蝇   |
| 7    | 褐飞虱     |
| 8    | 白背飞虱   |
| 9    | 灰飞虱     |

| 10   | 稻水象甲 |
| ---- | -------- |
| 11   | 稻叶蝉   |
| 12   | 稻蓟马   |
| 13   | 稻壳虫   |
| 14   | 蛴螬     |
| 15   | 蝼蛄     |
| 16   | 金针虫   |
| 17   | 白缘螟蛾 |
| 18   | 小地老虎 |
| 19   | 大地老虎 |

# Project structure

yolo_cv_detect/ 

├── backend/

│   ├── main.py          # FastAPI后端主程序 

│   ├── best.pt          # YOLO模型文件 

│   ├── static/          # 静态文件目录（存放前端HTML） 

│   │   	└── UI.html 

│   ├── uploads/         # 上传文件临时目录  初次运行时程序会自动创建

│   └── results/         # 检测结果保存目录  初次运行时程序会自动创建

└── requirements.txt     # 依赖清单

# Main Requirements

Ultralytics

Fastapi

Pillow,Numpy,wheel,scipy

websockets

pytorch

# Overall system architecture

该系统是一套基于**YOLO 模型** + **FastAPI 后端** + **HTML/CSS/JS 前端** 的农田害虫检测平台，采用前后端分离架构，支持图片 / 批量图片 / 视频 / 实时摄像头四种检测模式，核心架构如下：

|  层级  |        技术栈 / 核心文件         |                           核心作用                           |
| :----: | :------------------------------: | :----------------------------------------------------------: |
| 前端层 |       UI.html（静态页面）        | 提供可视化交互界面，支持文件上传、参数配置、结果展示、文件保存 |
| 后端层 |        main.py（FastAPI）        | 提供 API 接口、模型推理、文件管理、WebSocket 实时通信、结果生成 / 下载 |
| 模型层 |       YOLO（ultralytics）        | 基于预训练的 best.pt 模型实现害虫目标检测，输出类别、置信度、边界框 |
| 数据层 | 本地目录（uploads/results/temp） |   存储上传文件、检测结果、临时压缩包，支持文件持久化和下载   |

# Quick Start

大家需要先配置好环境，这里建议使用anaconda来配置虚拟环境

* 第一步需要先确定自己的IP地址

```cmd
#打开cmd，在cmd中输入以下指令查看本机IP地址
ipconfig
```
<img width="1056" height="251" alt="image-20260109203045404" src="https://github.com/user-attachments/assets/3ea2771b-1d04-4ca6-95bd-0e8907e04f44" />

查看并记录下本机的IP地址

* 第二步打开` main.py`后端文件

将配置项下` BASE_URL`中的IP地址替换成自己的IP地址

<img width="765" height="147" alt="image-20260109203208975" src="https://github.com/user-attachments/assets/3aa77c95-ae8e-41a1-8210-994f25d08a54" />

* 第三步生成 RSA 4096 位私钥 + 自签名 SSL/TLS 证书

  由于本项目是基于https协议下的，基于https的安全规则要求，要在服务器上部署 **HTTPS 协议**，**必须生成密钥对**（公钥 + 私钥），并配合由权威 CA（证书颁发机构）签名的 SSL/TLS 证书才能实现

  需要在cmd上运行` openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes`命令，该命令会**一次性完成了生成 RSA 4096 位私钥 + 自签名 SSL/TLS 证书**的全部操作，执行完成后终端会提示你填写证书的基本信息

  ```cmd
  Country Name (2 letter code) [AU]:CN
  State or Province Name (full name) [Some-State]:Beijing
  Locality Name (eg, city) []:Beijing
  Organization Name (eg, company) [Internet Widgits Pty Ltd]:MyTest
  Organizational Unit Name (eg, section) []:Test
  Common Name (e.g. server FQDN or YOUR name) []:localhost  # 关键！测试用填localhost，服务器用填域名
  Email Address []:test@example.com
  ```

  执行完成后，当前目录会生成两个文件：

  - `key.pem`：服务器私钥（绝对不能泄露，权限建议设为 `600`）
  - `cert.pem`：自签名证书（包含公钥，可公开）

  若没有更改文件路径，以上文件生成在用户目录下
  
<img width="1329" height="1428" alt="image-20260109204902484" src="https://github.com/user-attachments/assets/9110e33e-a390-4539-8ee3-d1d7ad1d8726" />

  若文件不在用户目录下，大家可自行修改
  
<img width="867" height="268" alt="image-20260109205224559" src="https://github.com/user-attachments/assets/2f43bbb6-6edb-4771-8865-28b94d4b1847" />

* 第四步运行` main.py`后端文件

看到如下情况则说明后端服务成功启动

<img width="1691" height="473" alt="image-20260109203416470" src="https://github.com/user-attachments/assets/fca52c15-0471-4c19-b0d4-b917a85db9f5" />

按下` ctrl + c`可退出后端服务

* 第五步接下就可以在同一个局域网内访问该程序

在浏览器中输入服务器的URL，就能够成功进入了！！！

<img width="1367" height="80" alt="image-20260109205525607" src="https://github.com/user-attachments/assets/360ae1bc-08f7-4938-b78e-9aaac016649d" />

<img width="1836" height="1449" alt="image-20260109205611343" src="https://github.com/user-attachments/assets/80041477-bdfe-4a12-afa3-2540cb3cbb46" />

# Thanking

这个是我们学校的视觉课程设计作业，在五天的课时勉强赶出来的，我知道这个小项目还存在诸多的问题，比如不会调整网络的结构，但是能实现基本的功能满足课程设计需求。

需要相关害虫数据集，或者有其他的问题可以联系这个邮箱` 790543331@qq.com`,我们会及时的恢复你

