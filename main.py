import streamlit as st
import cv2
import numpy as np

st.title("Blood Cell Detection")


with st.sidebar:
    CONFIDENCE_THRESHOLD = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)


# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.11
NMS_THRESHOLD = 0.45
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

BLACK  = (0,0,0)

COLORS = {1:(255, 0, 0),
          2: (0, 255, 0),
          0:(0, 0, 255)}

def draw_label(im, label, x, y,color):
    """Draw text onto image at location."""
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)



def pre_process(input_image, net):
      blob = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
      net.setInput(blob)
      outputs = net.forward(net.getUnconnectedOutLayersNames())
      return outputs


def post_process(input_image, outputs):
      class_ids = []
      confidences = []
      boxes = []
      rows = outputs[0].shape[1]
      image_height, image_width = input_image.shape[:2]
      x_factor = image_width / INPUT_WIDTH
      y_factor =  image_height / INPUT_HEIGHT
      for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            if confidence >= CONFIDENCE_THRESHOLD:
                  classes_scores = row[5:]
                  class_id = np.argmax(classes_scores)
                  if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)

      indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
      for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        color = COLORS[int((classes[class_ids[i]]).split(':')[0])]  
        cv2.rectangle(input_image, (left, top), (left + width, top + height), color, 1*THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top,color)
      return input_image


if __name__ == '__main__':
    classesFile = "models/coco.names"
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        buttton = st.button("start processsing")
        if buttton:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            modelWeights = "models/new_model.onnx"
            net = cv2.dnn.readNet(modelWeights)

            detections = pre_process(frame, net)
            img = post_process(frame.copy(), detections)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Processed Image", use_column_width=True)