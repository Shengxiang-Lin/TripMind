# TripMind 演示 POI 本地图片与署名

本次仅准备素材；没有修改 app、planner、recommender、service，也没有重建或改写 production bundle。

## 素材范围

- 杭州 16 个、北京 20 个 POI，每个一张本地 JPEG，共 36 张。
- 映射：`data/processed/images/poi_image_map.json`（项目根目录相对路径）。
- 依据 active bundle `cac6e997b10fd68f7952` 的 POI ID，不适用于原始杭州样例的 `hz...` ID。
- 先按每城 18 组画像的实际排程出现频率、标签及开放时间快照筛选，再人工核对地点与文件说明；并非真实人气榜。production 的 popularity 均为 0.5。
- 北京常见候选更集中于易辨识地标，18 组画像共选中 35 个不同 POI，上海为 36 个；最终选择杭州与北京。
- 选中照片覆盖抽样行程位置：杭州 43/108，北京 67/92；不是所有行程都保证有图。杭州茶博与动物园作为茶文化、亲子主题补充。
- 没有把邻近景点照片冒充目标 POI；南屏晚钟采用净慈寺钟楼铜钟，照片说明已在映射记录。
- 大兜路素材是街区内咖啡馆/街角细节，小河直街素材为历史水岸照片；均不是全景或实时街景保证。

## 来源与许可

所有素材来自下表列明的 Wikimedia Commons 文件页。使用 Commons 自身生成的约 960px 宽缩略图（竖图高度可能超过 960px），未做本地裁剪、修图或 AI 生成。原图地址、文件页固定修订链接、下载地址、拍摄日期、作者、许可链接、验证时间与 SHA-256 均保存于映射。
仅查询人工指定文件的元数据，不使用外部图片搜索 API。两张无机器可读作者的图片（三潭印月、北京动物园）已从说明页正文人工核对署名。
图片采用各自的 CC BY / CC BY-SA、CC0 或 Public domain 声明，不应统一标成仓库代码许可证。后续 Gallery 应保留作者、来源和许可链接；对 CC BY-SA 图片的改编应遵守其同方式共享条款。
照片为各自拍摄日期的历史记录；开放时间仅为原有静态数据，未做当前营业核实。

## Windows 文件命名

原始 poi_id 含有冒号，不能直接作为 Windows 文件名。将 `%` 编为 `%25`、`:` 编为 `%3A`，再加 `.jpg`；中文保持不变。例：`chinatravel:杭州:35` → `chinatravel%3A杭州%3A35.jpg`。映射的 poi_id 保持原值，可对文件名 stem 作 URL 解码恢复。

## 逐图署名清单

| 城市 | POI / ID | 作者 | 本次采用的许可 | 来源 |
| --- | --- | --- | --- | --- |
| 杭州 | 龙井村 / `chinatravel:杭州:35` | Windmemories | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:20260424_Longjing_Village.jpg) |
| 杭州 | 雷峰塔 / `chinatravel:杭州:9` | lienyuan lee | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Leifeng_Pagoda_%E9%9B%B7%E5%B3%B0%E5%A1%94_-_panoramio.jpg) |
| 杭州 | 杭州博物馆 / `chinatravel:杭州:57` | 猫猫的日记本 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Hangzhou_Museum_01_2018-07.jpg) |
| 杭州 | 南宋御街 / `chinatravel:杭州:91` | 钉钉 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Southern_Song_Imperial_Street.jpg) |
| 杭州 | 南屏晚钟 / `chinatravel:杭州:150` | Tyg728 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:%E5%87%80%E6%85%88%E5%AF%BA%E9%92%9F%E6%A5%BC%E9%93%9C%E9%92%9F.jpg) |
| 杭州 | 中国茶叶博物馆 / `chinatravel:杭州:202` | 三猎 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:%E4%B8%AD%E5%9B%BD%E8%8C%B6%E5%8F%B6%E5%8D%9A%E7%89%A9%E9%A6%861.jpg) |
| 杭州 | 清河坊街 / `chinatravel:杭州:20` | HoweyYuan | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Hefang_Street_20240729.jpg) |
| 杭州 | 小河直街历史文化街区 / `chinatravel:杭州:46` | zhiyin586@163.com | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:%E6%9D%AD%E5%B7%9E._%E8%BF%90%E6%B2%B3%E6%B0%B4%E4%B8%8A%E5%B7%B4%E5%A3%AB_%EF%BC%88%E5%B0%8F%E6%B2%B3%E7%9B%B4%E8%A1%97%EF%BC%89_-_panoramio.jpg) |
| 杭州 | 大兜路历史文化街区 / `chinatravel:杭州:122` | zhiyin586@163.com | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:%E6%9D%AD%E5%B7%9E._%E5%A4%A7%E5%85%9C%E8%B7%AF%E5%8E%86%E5%8F%B2%E8%A1%97%E5%8C%BA_-_panoramio.jpg) |
| 杭州 | 杭州动物园 / `chinatravel:杭州:18` | Huandy618 | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Hangzhou_Zoo_03.jpg) |
| 杭州 | 西湖风景名胜区 / `chinatravel:杭州:2` | Louisa Salazar | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:West_Lake_-_Hangzhou,_China.jpg) |
| 杭州 | 钱塘江 / `chinatravel:杭州:10` | Windmemories | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:20260802_Qiantang_River_in_Hangzhou.jpg) |
| 杭州 | 三潭印月 / `chinatravel:杭州:16` | Shizhao | [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Three_Pools_Mirroring_the_Moon-pool.JPG) |
| 杭州 | 苏堤 / `chinatravel:杭州:55` | Yumeto | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:20250715_Su_Di.jpg) |
| 杭州 | 太子湾公园 / `chinatravel:杭州:74` | Huandy618 | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Taiziwan_Park_01.jpg) |
| 杭州 | 灵隐寺 / `chinatravel:杭州:0` | Denis Yuen | [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Lingyin_si.JPG) |
| 北京 | 雍和宫 / `chinatravel:北京:14` | Shakti | [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Pavillon_des_dix_mille_bonheurs.jpg) |
| 北京 | 大栅栏 / `chinatravel:北京:59` | WhisperToMe | [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Dazhalan.JPG) |
| 北京 | 颐和园 / `chinatravel:北京:4` | Zhangzhe0101 | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Scenery_of_Longevity_Hill.JPG) |
| 北京 | 圆明园 / `chinatravel:北京:7` | 颐园新居 | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Yuanmingyuan_Haiyantang_20130126.JPG) |
| 北京 | 北京天文馆 / `chinatravel:北京:25` | Shizhao | [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Beijing_Planetarium_new.jpg) |
| 北京 | 首都博物馆 / `chinatravel:北京:75` | PENG Yanan (Neo-Jay/鸟甲) | [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Capital_Museum_in_Beijing.jpg) |
| 北京 | 西单商业街 / `chinatravel:北京:114` | N509FZ | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Xidan_Pailou_(20211007134510).jpg) |
| 北京 | 故宫博物院 / `chinatravel:北京:1` | user:kallgan | [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Forbidden_City_Beijing_Shenwumen_Gate.JPG) |
| 北京 | 天安门广场 / `chinatravel:北京:13` | Paul Louis | [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Tiananmen_Square.JPG) |
| 北京 | 南锣鼓巷 / `chinatravel:北京:29` | Dimitris Argyris from Athens, Greece. Edited slightly prior to upload by Daniel Case | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:View_down_Nanluoguxiang_2011.jpg) |
| 北京 | 午门 / `chinatravel:北京:47` | そらみみ | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Meridian_Gate_of_the_Forbidden_City.jpg) |
| 北京 | 清华大学艺术博物馆 / `chinatravel:北京:50` | HoweyYuan | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Tsinghua_University_Art_Museum_Huang_Rulun_Building_(20240707145554).jpg) |
| 北京 | 国家图书馆 / `chinatravel:北京:81` | N509FZ | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:North_wing_of_the_National_Library_of_China_(20230318082607).jpg) |
| 北京 | 长安街 / `chinatravel:北京:154` | Lkjidm | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:2021%E4%BA%94%E4%B8%80%E5%81%87%E6%9C%9F%E9%95%BF%E5%AE%89%E8%A1%97.jpg) |
| 北京 | 北京动物园 / `chinatravel:北京:9` | Snowyowls | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Front_Gate_of_Beijing_Zoo.JPG) |
| 北京 | 北海公园 / `chinatravel:北京:28` | me (w:User:pfctdayelise) | [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Beihai_park_-_bridge_to_white_pagoda.JPG) |
| 北京 | 前门大街 / `chinatravel:北京:37` | WiNG | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Qianmen_Dajie_View1_2010.jpg) |
| 北京 | 王府井 / `chinatravel:北京:38` | EditQ | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:%E7%8E%8B%E5%BA%9C%E4%BA%95%E5%A4%A7%E8%A1%97.jpg) |
| 北京 | 天安门城楼 / `chinatravel:北京:39` | Charlie fong; 冯成 | [Public domain](https://creativecommons.org/publicdomain/mark/1.0/) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:%E5%A4%A9%E5%AE%89%E9%97%A8%E5%A4%9C%E6%99%AF.jpg) |
| 北京 | 国子监 / `chinatravel:北京:65` | Peiyu Liu | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0) | [Commons 文件页](https://commons.wikimedia.org/wiki/File:Guozijian_building_1.jpg) |

## 校验

已逐张检查 JPEG 解码、像素尺寸、文件长度及 SHA-256，并以本地拼图核对图片内容。映射仅包含存在的本地文件；不依赖运行时联网下载。
production current.json 与 manifest.json 的 SHA-256 在素材准备前后核对，未改变。
素材和映射均已纳入 Git；映射所在 processed 目录原本被忽略，本文件采用显式强制添加，未放开其他生成数据的忽略规则。
