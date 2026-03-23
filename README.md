# 人脸识别系统 (HW03)

基于 `face_recognition` 和 `Streamlit` 的人脸识别系统，支持人脸检测、特征提取和身份识别。

## 功能特点

- **人脸检测**：自动检测图片中所有人脸位置
- **人脸特征提取**：生成 128 维人脸特征编码
- **人脸识别**：与已知人脸库比对，识别身份
- **Web 界面**：基于 Streamlit 的交互式界面
- **实时标注**：在图片上绘制人脸框和标签

## 项目结构
hw03/
├── app.py # Streamlit 主程序
├── requirements.txt # Python 依赖
├── README.md # 项目说明
├── src/
│ ├── init.py
│ └── face_utils.py # 人脸识别核心逻辑
├── tests/
│ └── test_face_utils.py # 单元测试
└── images/ # 示例图片目录
└── sample.jpg
