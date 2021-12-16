from django.shortcuts import render
import os
# Build paths inside the project like this: BASE_DIR / 'subdir'.
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
from PIL import Image
import warnings
from object_detection.utils.visualization_utils import count_label

def index(request):
    if request.method == 'POST' and 'image' in request.FILES:
        imgs = request.FILES['image']
        im = Image.open(imgs)
        im.show()
        savepath = "/static/images"
        print(type(imgs))
        warnings.filterwarnings('ignore')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        def download_images():
            image_paths = [imgs]
            return image_paths

        IMAGE_PATHS = download_images()

        def download_model():
            dirs = BASE_DIR.joinpath('custom_models/centernet_hg104_1024x1024_coco17_tpu-32')
            return str(dirs)

        # MODEL_DATE = '20200711'
        # MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
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
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            a = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

            save_img = Image.fromarray(image_np_with_detections)
            save_img.save(str(BASE_DIR)+"/static/images/" + str(IMAGE_PATHS[0]))
            print(a[1], "countings")
        return render(request,"index.html", {'predict': a[1],'image':IMAGE_PATHS[0]})
    else:
        return render(request, "index.html")