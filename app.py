"""
人脸识别系统 - Streamlit Web 界面
基于 face_recognition 和 Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

from src.face_utils import FaceRecognizer

# 页面配置
st.set_page_config(
    page_title="人脸识别系统",
    page_icon="👤",
    layout="wide"
)

# 初始化 Session State
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = FaceRecognizer()
if 'known_faces' not in st.session_state:
    st.session_state.known_faces = {}
if 'known_names' not in st.session_state:
    st.session_state.known_names = []


def load_known_faces_from_folder(folder_path):
    """从文件夹加载已知人脸库"""
    if not os.path.exists(folder_path):
        return
    
    recognizer = st.session_state.recognizer
    st.session_state.known_faces.clear()
    st.session_state.known_names.clear()
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            filepath = os.path.join(folder_path, filename)
            image = cv2.imread(filepath)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encoding = recognizer.get_face_encoding(image_rgb)
                if encoding is not None:
                    st.session_state.known_faces[name] = encoding
                    st.session_state.known_names.append(name)
    
    if st.session_state.known_faces:
        st.success(f"已加载 {len(st.session_state.known_faces)} 个已知人脸")


def main():
    # 标题
    st.title("👤 人脸识别系统")
    st.markdown("基于 **face_recognition** 和 **Streamlit** 的人脸检测与识别系统")
    st.divider()
    
    # 侧边栏 - 配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # 人脸库管理
        st.subheader("📁 人脸库管理")
        known_faces_path = st.text_input(
            "已知人脸库文件夹路径",
            placeholder="例如: ./known_faces/",
            help="将已知人脸的图片放入该文件夹，文件名将作为人名"
        )
        
        if known_faces_path and st.button("加载人脸库"):
            load_known_faces_from_folder(known_faces_path)
        
        if st.session_state.known_faces:
            st.write(f"已加载 {len(st.session_state.known_faces)} 个人脸:")
            for name in st.session_state.known_names:
                st.write(f"- {name}")
        
        st.divider()
        
        # 识别参数
        st.subheader("🎛️ 识别参数")
        tolerance = st.slider(
            "匹配阈值 (tolerance)",
            min_value=0.3,
            max_value=0.8,
            value=0.6,
            step=0.01,
            help="值越小匹配越严格"
        )
        
        # 示例图片
        st.subheader("📷 示例图片")
        use_example = st.checkbox("使用示例图片")
        if use_example:
            example_image_path = "images/sample.jpg"
            if os.path.exists(example_image_path):
                st.image(example_image_path, caption="示例图片", width=200)
            else:
                st.warning("示例图片不存在，请将图片放入 images/ 目录")
    
    # 主区域 - 图片上传和处理
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 上传图片")
        
        # 图片上传
        uploaded_file = st.file_uploader(
            "选择一张图片",
            type=["jpg", "jpeg", "png"],
            help="支持 JPG、JPEG、PNG 格式"
        )
        
        # 显示上传的图片
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_container_width=True)
            
            # 保存到 session
            st.session_state.uploaded_image = np.array(image)
        
        elif use_example and os.path.exists("images/sample.jpg"):
            example_img = Image.open("images/sample.jpg")
            st.image(example_img, caption="示例图片", use_container_width=True)
            st.session_state.uploaded_image = np.array(example_img)
        
        else:
            st.info("👈 请上传图片或选择示例图片")
    
    with col2:
        st.header("🔍 识别结果")
        
        # 处理按钮
        if st.button("开始识别", type="primary", use_container_width=True):
            if 'uploaded_image' not in st.session_state:
                st.warning("请先上传图片")
            else:
                with st.spinner("正在处理中..."):
                    recognizer = st.session_state.recognizer
                    image = st.session_state.uploaded_image
                    
                    # 检测人脸位置
                    face_locations = recognizer.detect_faces(image)
                    
                    if not face_locations:
                        st.warning("未检测到人脸")
                    else:
                        # 获取人脸特征编码
                        face_encodings = recognizer.get_face_encodings(image, face_locations)
                        
                        # 识别结果
                        results = []
                        for i, encoding in enumerate(face_encodings):
                            name = "Unknown"
                            if st.session_state.known_faces:
                                name, distance = recognizer.recognize_face(
                                    encoding,
                                    st.session_state.known_faces,
                                    tolerance
                                )
                                results.append({
                                    "location": face_locations[i],
                                    "name": name,
                                    "distance": distance if name != "Unknown" else None
                                })
                            else:
                                results.append({
                                    "location": face_locations[i],
                                    "name": f"Person {i+1}",
                                    "distance": None
                                })
                        
                        # 在图片上标注
                        annotated_image = recognizer.draw_face_boxes(
                            image.copy(),
                            face_locations,
                            [r["name"] for r in results]
                        )
                        
                        # 显示结果
                        st.image(annotated_image, caption="识别结果", use_container_width=True)
                        
                        # 显示详细信息
                        st.subheader("📋 检测详情")
                        for i, result in enumerate(results):
                            location = result["location"]
                            st.write(f"**人脸 {i+1}**: {result['name']}")
                            st.write(f"  - 位置: top={location[0]}, right={location[1]}, bottom={location[2]}, left={location[3]}")
                            if result["distance"]:
                                st.write(f"  - 匹配距离: {result['distance']:.4f}")
                            st.divider()
        
        # 清除按钮
        if st.button("清除", use_container_width=True):
            if 'uploaded_image' in st.session_state:
                del st.session_state.uploaded_image
            st.rerun()


if __name__ == "__main__":
    main()
