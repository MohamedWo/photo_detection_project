
# import cv2

# # تحميل الصورة
# img = cv2.imread('ww/people-6027028_1280.jpg')

# # تحميل أسماء الكائنات
# classname = []
# classfiles = "qq/Things.name"

# with open(classfiles, "r") as f:
#     classname = f.read().rstrip('\n').split('\n')

# # تحميل النموذج
# p = "qq/frozen_inference_graph.pb"
# v = "qq/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# net = cv2.dnn_DetectionModel(p, v)
# net.setInputSize(320, 320)
# net.setInputScale(1.0 / 127.5)
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)

# # اكتشاف الكائنات
# classIds, confs, bbox = net.detect(img, confThreshold=0.5)

# print("Class IDs:", classIds)
# print("Bounding Boxes:", bbox)

# # رسم المربعات على الصورة
# if len(classIds) != 0:
#     for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
#         cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

#         # التأكد أن classId داخل حدود عدد الأصناف
#         if classId <= len(classname):
#             label = f"{classname[classId-1].upper()} {round(confidence*100,2)}%"
#         else:
#             label = f"UNKNOWN {round(confidence*100,2)}%"

#         cv2.putText(img, label, (box[0]+10, box[1]+30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# # عرض الصورة
# cv2.imshow("Detections", img)

# cv2.waitKey(0)






import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Object Detection باستخدام SSD MobileNet")

# رفع الصورة
uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # تحويل الصورة إلى صيغة يمكن لـ OpenCV التعامل معها
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # تحميل أسماء الكائنات
    classfiles = "qq/Things.name"
    with open(classfiles, "r") as f:
        classname = f.read().rstrip('\n').split('\n')

    # تحميل النموذج
    p = "qq/frozen_inference_graph.pb"
    v = "qq/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

    net = cv2.dnn_DetectionModel(p, v)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # اكتشاف الكائنات
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    # رسم المربعات على الصورة
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

            if classId <= len(classname):
                label = f"{classname[classId-1].upper()} {round(confidence*100,2)}%"
            else:
                label = f"UNKNOWN {round(confidence*100,2)}%"

            cv2.putText(img, label, (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # تحويل الصورة إلى PIL لعرضها في Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(Image.fromarray(img_rgb), caption='الصورة بعد الكشف عن الكائنات', use_column_width=True)
