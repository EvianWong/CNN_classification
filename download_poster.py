import pandas as pd
import requests

n = 1                           # 下载图片数量

data = pd.read_csv('data/MovieGenre.csv', encoding='gbk')
data.dropna(inplace=True)

img_web = data['Poster']
img_title = data['Title']
genre = data['Genre']
index = data.index

post_index = []
for value in index[0:n]:
    try:
        response = requests.get(img_web[value])
        if response.content == b'Not Found':
            continue
        else:
            with open('data/poster/' + str(value) + '.jpg', 'wb') as f:
                f.write(response.content)
                post_index.append(value)
    except:
        continue

# 成功保存的海报索引
test = pd.DataFrame(columns=['post_index'], data=post_index)
test.to_csv('data/post_index2.csv', encoding='utf-8')
