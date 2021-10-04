import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import numpy as np
from PIL import Image

import cv2
import numpy as np

class detector():
    def __init__(self):
        CUSTOM_MODEL_NAME = 'my_centernet_resnet50_v2' 
        PRETRAINED_MODEL_NAME = 'centernet_resnet50_v2_512x512_coco17_tpu-8'
        PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz'
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'
        TF_ws = 'TensorFlow_ws'

        paths = {
            'WORKSPACE_PATH': os.path.join(TF_ws , 'workspace'),
            'SCRIPTS_PATH': os.path.join(TF_ws ,'scripts/preprocessing'),
            'APIMODEL_PATH': os.path.join(TF_ws ,'models'),
            'ANNOTATION_PATH': os.path.join(TF_ws , 'workspace','training_demo','annotations'),
            'IMAGE_PATH': os.path.join(TF_ws , 'workspace','training_demo','images'),
            'MODEL_PATH': os.path.join(TF_ws , 'workspace','training_demo','models'),
            'PRETRAINED_MODEL_PATH': os.path.join(TF_ws , 'workspace','training_demo','pre-trained-models'),
            'CHECKPOINT_PATH': os.path.join(TF_ws , 'workspace','training_demo','models',CUSTOM_MODEL_NAME), 
            'OUTPUT_PATH': os.path.join(TF_ws , 'workspace','training_demo','exported-models',CUSTOM_MODEL_NAME), 
            'PROTOC_PATH':os.path.join('protoc')
        }

        files = {
            'PIPELINE_CONFIG':os.path.join(TF_ws, 'workspace','training_demo','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
            'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
            'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
        }

        PATH_TO_MODEL_DIR = paths['OUTPUT_PATH']
        PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

        self.category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'],use_display_name=True)
        self.detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    def load_image_into_numpy_array(path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
        path: the file path to the image

        Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(Image.open(path))

    # print out what are detected!!

    # Load pipeline config and build a detection model
    def load_model_from_ckpt():
        '''configs = config_util.get_configs_from_pipeline_file(PATH_TO_MODEL_DIR+'/pipeline.config')
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'],'checkpoint','ckpt-0')).expect_partial()
        '''

    # print out what are detected!!
    def print_detect_detials(self,detections, threshold=0.5):
        for i in range(detections['num_detections']):
            detect_score = detections['detection_scores'][i]
            index_class = detections['detection_classes'][i]

            if detect_score > threshold:
                print(f"{self.category_index[index_class]['name']} , score = {detect_score*100:.2f} %")

    def video_detection(self):

        cap = cv2.VideoCapture(0)
        while cap.isOpened(): 
            ret, frame = cap.read()
            image_np = np.array(frame)

            input_tensor = tf.convert_to_tensor(image_np,dtype=tf.uint8)
            input_tensor = input_tensor[tf.newaxis, ...]
            detections = self.detect_fn(input_tensor)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = image_np.copy()

            self.print_detect_detials(detections,threshold=0.5)

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image = image_np_with_detections,
                boxes = detections['detection_boxes'],
                classes = detections['detection_classes'],
                scores = detections['detection_scores'],
                category_index = self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=0.5,
                agnostic_mode=False)

            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    x = detector()
    x.video_detection()