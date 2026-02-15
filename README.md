# F045 vue+flask棉花病虫害CNN识别+AI问答知识neo4j 图谱可视化系统深度学习神经网络

完整项目收费，可联系QQ: 81040295 微信: mmdsj186011 注明从github来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775

>关注B站，有好处！
>B站号： 麦麦大数据
# 功能介绍
 编号：F045
 vue+flask+neo4j+mysql 架构 （前后端分离架构）
 棉花医院AI问答：前端聊天界面体验超棒（对接千问大模型API）
 病虫害图片识别：基于CNN的棉花病虫害识别，可自己训练模型（基于pytorch）
 数据为棉花的四个期对应的 各种虫害
 知识图谱： 模糊查询+图标+双击+拖动等（双击展示数据）
 数据大屏：中国地图显示产地： echarts 分析
 病虫害查询： 分页+模糊查询+卡片展示
 棉花生长周期：展示4个周期
 关键词分析等、登录注册
# 1 视频讲解

[video(video-XxgeFcYP-1732246616269)(type-bilibili)(url-https://player.bilibili.com/player.html?aid=113099591909862)(image-https://img-blog.csdnimg.cn/img_convert/3be77bfbabcd34470c5a518da83e5551.jpeg)(title-F045vue+flask棉花病虫害CNN识别+AI问答知识neo4j 图谱可视化系统深度学习神经网络)]

# 2 架构
## 功能简介
本系统是一个基于Vue+Flask构建的棉花病虫害AI识别与知识图谱可视化系统，旨在为棉花种植提供智能化的病虫害识别、AI问答以及知识管理服务。系统的核心功能模块包括：AI问答系统，提供与千问大模型对接的智能问答服务，支持用户就棉花病虫害相关问题进行提问；病虫害图片识别模块，基于CNN深度学习模型实现棉花病虫害的图像识别，并支持模型的二次训练功能；知识图谱模块，通过Neo4j实现病虫害知识的存储、查询与可视化，支持模糊查询、双击节点展示详情、图谱拖动等交互功能；数据可视化模块，利用ECharts展示棉花病虫害数据的大屏分析，包括中国地图产地分布、关键词分析以及词云生成；病虫害查询模块，支持分页浏览、模糊查询以及卡片式数据展示；棉花生长周期模块，直观展示棉花的四个生长周期相关信息；用户管理模块，提供身份验证、信息修改、头像设置、密码找回（支持短信验证）以及身份证OCR识别等功能，确保系统的安全性和用户体验。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/da931ca3b61147eaa1b6dfbb637afaa6.png)
## 架构说明
该系统采用B/S（浏览器/服务器）架构模式，前端基于Vue.js框架构建，集成了Vue Router（路由管理）、Vuex（状态管理）、ECharts（数据可视化）等技术，提供流畅的用户交互体验。后端采用Flask框架，负责业务逻辑处理与API接口的搭建，并通过MySQL数据库实现系统数据的持久化存储。知识图谱功能模块采用Neo4j数据库，用于存储和管理棉花病虫害相关的知识实体及其关联关系。病虫害图片识别模块基于PyTorch深度学习框架构建，支持CNN模型的训练与部署，能够识别棉花生长周期中不同的病虫害类型。AI问答功能对接千问大模型API，提供智能化的问答服务。系统还集成了自然语言处理技术，采用jieba、TF-IDF和TextRank算法进行关键词提取和词云分析，为用户提供文本数据的深度洞察。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2d29ee8de8aa48f7bde12b515235f5c8.png)
# 3 病虫害知识问答
基于阿里千问大模型API实现的棉花病虫害问答
## 类似聊天界面
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49078b3e90284451b38cb85a71303e0f.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b111ce98f5474ff18a24d23c0b239c2a.jpeg)
# 4 病虫害识别
基于Pytorch CNN卷积神经网络模型实现的病虫害识别
识别叶甲
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d68d3a971abd4b29824838bc80baf40a.jpeg)
上传图片，右侧展示识别结果、图片和相关的信息：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/100d6a3621c8468581eeb2a6277972c7.jpeg)
# 5 知识图谱可视化
## 图谱的导入
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/764fe9d562a14ac7a4ceb4e91d91e9ee.png)
## neo4j 界面
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/67ad00b43c3f46e89ff9b422f88a11b3.png)
## 可视化
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6244442462ee4d99840271b3ca121e9b.jpeg)
## 支持模糊搜索显示知识图谱子图，输入“蕾”
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae3e1d98899f45239b7e2c98da31e345.jpeg)
## 点击节点，右侧展示详细节点信息
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a09ac81a78db43dfa0a171c746698b35.jpeg)
# 6 病虫害知识库搜索
可以搜索各种病虫害，支持模糊搜索+分页，画面美观
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0633d0149a5842b0a8d23b9b4be39f16.jpeg)
# 7 棉花周期科普
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8d7cfb2bbbef437382ae0f64a44a6f02.jpeg)
# 8 关键词分析
基于统计、textrank+tfidf双算法的关键词主题词分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c88b8476f12e4c8cbcdb67736de75cf5.jpeg)
# 9 词云分析
基于jieba分词的词云分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b366b689843342898367d586a58a8112.jpeg)
# 10 数据大屏
多种echarts可视化图形数据分析的应用，美观大方
通过中国地图分析棉花产地、药物类型、虫害分析、有效成分等
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/410ed1d3678c46b4b654a92e7fa69a23.jpeg)
# 11 登录和注册
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/31f1508b87e64e2eb141a65983657b94.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/84e4fd65110e447a84925637a781bb07.jpeg)
# 12 个人信息设置，可修改头像等
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/054c9b2c4e8c4308b2f4b7c4f091e828.jpeg)

# 13 病虫害识别代码
## 功能说明
该代码实现了一个基于CNN的棉花病虫害识别系统，使用TensorFlow和Keras框架。首先，代码导入必要的库，并准备数据集，包括训练集和验证集。通过ImageDataGenerator进行数据增强和预处理，以增加模型的泛化能力。随后，代码加载了预训练的VGG16模型，并在其基础上添加了自定义的分类层，构建了适用于棉花病虫害识别的CNN模型。模型通过Adam优化器和交叉熵损失函数进行训练，并在验证集上评估性能。最终，模型保存为H5格式，方便后续部署和使用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/07f0b5e119ef444a817c0b63e1128be8.png)

## 核心代码
```python 

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 预测函数
def predict(image_path, model, class_names):
    # 定义图像预处理
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),  # 统一大小
    #     transforms.ToTensor(),
    # ])

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Lambda(lambda x: x.convert('RGB')),  # 确保转换为RGB模式
        transforms.ToTensor(),
    ])

    # 加载和预处理图像
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 增加批次维度

    # 将图像输入模型进行预测
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        outputs = model(image)
        # print(outputs)
        _, predicted = torch.max(outputs, 1)
    print(predicted.item())
    # 返回预测的类别
    return class_names[predicted.item()]

def predict_interface(test_image_path):
    # 加载训练好的模型
    num_classes = 5  # 根据你的数据集类别数量修改
    model = SimpleCNN(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 类别名称（根据你的数据集修改）
    class_names = ['中黑盲蝽', '台龟甲', '叶甲', '宽棘缘椿', '小长蝽',
                   ]  # 替换为实际类别名称
    # 测试预测
    # test_image_path = '3.jpg'  # 替换为测试图像的路径
    predicted_class = predict(test_image_path, model, class_names)
    return predicted_class

if __name__ == "__main__":
    # 加载训练好的模型
    num_classes = 5  # 根据你的数据集类别数量修改
    model = SimpleCNN(num_classes)
    model.load_state_dict(torch.load('disease_model.pth'))
    model.eval()

    # 类别名称（根据你的数据集修改）
    class_names = ['中黑盲蝽', '台龟甲', '叶甲', '宽棘缘椿', '小长蝽',
]  # 替换为实际类别名称

    # 测试预测
    test_image_path = '3.jpg'  # 替换为测试图像的路径
    predicted_class = predict(test_image_path, model, class_names)
    print(f'Predicted class: {predicted_class}')
```
