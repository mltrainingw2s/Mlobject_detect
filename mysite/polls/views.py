from django.shortcuts import render

# Create your views here.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import pathlib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder         
import numpy as np
import matplotlib.pyplot as plt
import warnings
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  
from pathlib import Path
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image
from rest_framework.decorators import api_view
gpus = tf.config.experimental.list_physical_devices('GPU')
import copy
print("gpus",gpus)
BASE_DIR = Path(__file__).resolve().parent.parent
print("BASE_DIR",BASE_DIR)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
@api_view(['GET', 'POST'])
def get_all(request):
    if request.method == "POST":
        data = request.data
        file = request.FILES["imageFile"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.size(file_name)
        print("file",file_url)
        def download_images():
            base_url = "D:/django-web/mysite/media/" + str(data['imageFile'])
            image_paths = []
            image_paths.append(base_url)
            return image_paths
        IMAGE_PATHS = download_images()
        def download_model():
            print("BASE_DIR",BASE_DIR)
            model_dir=BASE_DIR.joinpath("custom_models/centernet_hg104_1024x1024_coco17_tpu-32")
            return str(model_dir)
        PATH_TO_MODEL_DIR = download_model()
        def download_labels(filename):
            label_dir = BASE_DIR.joinpath("custom_models/mscoco_label_map.pbtxt")
            return str(label_dir)
        LABEL_FILENAME = 'mscoco_label_map.pbtxt'
        PATH_TO_LABELS = download_labels(LABEL_FILENAME)
        PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
        PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"
        start_time = time.time()
        configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()
        @tf.function
        def detect_fn(image):
            """Detect objects in image."""
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            print("detections",detections)
            return detections
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                            use_display_name=True)
        def load_image_into_numpy_array(path):
            return np.array(Image.open(path))


        for image_path in IMAGE_PATHS:

            print('Running inference for {}... '.format(image_path), end='')

            image_np = load_image_into_numpy_array(image_path)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            print("sa",num_detections)
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections
            print("detection",detections['detection_classes'])
            # detection_classes should be ints.
            # print("detections['detection_classes']",detections['detection_classes'])
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            # print("detections['detection_classes']",detections['detection_classes'] + label_id_offset)
            a = viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=.30,
                    agnostic_mode=False)
            # print("a")
            print("IMAGE_PATHS",IMAGE_PATHS[0].split('/')[-1])
            array = Image.fromarray(image_np_with_detections)
            print(":arr",array)
            # print("a",a_)
            a_ = str(BASE_DIR)
            t = a_.split('\\')
            q = '/'
            s = q.join(t)
            array.save(s+'/static/img/'+IMAGE_PATHS[0].split('/')[-1])
            # array.save('D:/django-web/mysite/static/img/'+IMAGE_PATHS[0].split('/')[-1])
            return render(request,"index.html",{"image":IMAGE_PATHS[0].split('/')[-1],"number":a[2]})
    else:
        a =10
        print("a",a)
        b = [1,2,3]
        d = b.copy()
        # print()
        print("b",d)
        a = d.pop()
        print("test",a)
        print("d",b)
        return render(request, "index.html")
