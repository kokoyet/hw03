"""
人脸识别核心工具模块
包含人脸检测、特征提取、识别等功能
"""

import face_recognition
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class FaceRecognizer:
    """人脸识别器类"""
    
    def __init__(self):
        """初始化人脸识别器"""
        pass
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图片中所有人脸的位置
        
        Args:
            image: RGB 格式的图片数组 (height, width, 3)
        
        Returns:
            人脸位置列表，每个元素为 (top, right, bottom, left)
        """
        # 确保图片是 RGB 格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 已经是 RGB
            rgb_image = image
        else:
            # 转换到 RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测人脸位置
        face_locations = face_recognition.face_locations(rgb_image)
        return face_locations
    
    def get_face_encodings(
        self, 
        image: np.ndarray, 
        face_locations: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[np.ndarray]:
        """
        获取人脸的128维特征编码
        
        Args:
            image: RGB 格式的图片数组
            face_locations: 人脸位置列表，若为 None 则自动检测
        
        Returns:
            人脸特征编码列表
        """
        # 确保图片是 RGB 格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if face_locations is None:
            face_locations = self.detect_faces(rgb_image)
        
        if not face_locations:
            return []
        
        # 获取特征编码
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        return face_encodings
    
    def get_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        获取单张人脸的特征编码（假设图片中只有一张人脸）
        
        Args:
            image: RGB 格式的图片数组
        
        Returns:
            人脸特征编码，若未检测到人脸则返回 None
        """
        encodings = self.get_face_encodings(image)
        return encodings[0] if encodings else None
    
    def compare_faces(
        self,
        unknown_encoding: np.ndarray,
        known_encodings: Dict[str, np.ndarray],
        tolerance: float = 0.6
    ) -> Dict[str, float]:
        """
        将未知人脸与已知人脸库比对
        
        Args:
            unknown_encoding: 未知人脸的特征编码
            known_encodings: 已知人脸字典 {name: encoding}
            tolerance: 匹配阈值，越小越严格
        
        Returns:
            匹配结果字典 {name: distance}，距离越小越相似
        """
        results = {}
        for name, known_encoding in known_encodings.items():
            # 计算欧氏距离
            distance = np.linalg.norm(unknown_encoding - known_encoding)
            if distance <= tolerance:
                results[name] = distance
        
        # 按距离排序
        return dict(sorted(results.items(), key=lambda x: x[1]))
    
    def recognize_face(
        self,
        unknown_encoding: np.ndarray,
        known_encodings: Dict[str, np.ndarray],
        tolerance: float = 0.6
    ) -> Tuple[str, Optional[float]]:
        """
        识别单张人脸
        
        Args:
            unknown_encoding: 未知人脸的特征编码
            known_encodings: 已知人脸字典
            tolerance: 匹配阈值
        
        Returns:
            (识别结果名称, 最小距离)，若未识别则返回 ("Unknown", None)
        """
        matches = self.compare_faces(unknown_encoding, known_encodings, tolerance)
        
        if matches:
            # 返回匹配度最高的人脸
            best_match = list(matches.items())[0]
            return best_match[0], best_match[1]
        
        return "Unknown", None
    
    def draw_face_boxes(
        self,
        image: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        names: List[str] = None,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        在图片上绘制人脸框和标签
        
        Args:
            image: 图片数组
            face_locations: 人脸位置列表
            names: 对应的人名列表
            color: 边框颜色 (BGR)
        
        Returns:
            标注后的图片
        """
        # 确保是 BGR 格式用于 OpenCV 绘制
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 可能是 RGB，需要转换
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image.copy()
        
        if names is None:
            names = [f"Face {i+1}" for i in range(len(face_locations))]
        
        for (top, right, bottom, left), name in zip(face_locations, names):
            # 绘制边框
            cv2.rectangle(bgr_image, (left, top), (right, bottom), color, 2)
            
            # 绘制背景和文字
            label = name
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = top - 10 if top - 10 > 10 else top + 10
            
            cv2.rectangle(
                bgr_image,
                (left, label_y - label_size[1] - 5),
                (left + label_size[0] + 5, label_y + 5),
                color,
                -1
            )
            cv2.putText(
                bgr_image,
                label,
                (left + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # 转换回 RGB 用于 Streamlit 显示
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
