import mediapipe as mp
import random
import cv2
import time


class Image:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        self.right_eye_cords = [4, 5, 6, 8, 9, 261, 266, 275, 276, 279, 280, 282, 283, 293, 296, 334, 336, 342, 346,
                                353, 440, 265, 300, 301, 345]
        self.left_eye_cords = [3, 4, 5, 6, 8, 9, 35, 45, 46, 47, 63, 65, 66, 68, 71, 105, 107,
                               111, 131, 197, 36, 49, 50]

        self.left_iris_cords = [474, 475, 476, 477]
        self.right_iris_cords = [469, 470, 471, 472]

        self._face_init = None
        self._binary_face_init = None
        self._labels_init = None

        self.left_eye, self.right_eye, self.face, self.labels = [], [], [], []
        self.correct_idx = []

        self.init_size = None
        self.size = None
        self.mirror = None
        self.YCrCb = False
        self.data_index = []

    def _crop_wrap(self, img, box, show: str = None):
        sides = sorted([np.sum(abs(box[0] - box[1])).astype(int),
                        np.sum(abs(box[1] - box[2])).astype(int),
                        np.sum(abs(box[2] - box[3])).astype(int),
                        np.sum(abs(box[3] - box[0])).astype(int)])
        new_height = sides[3]
        new_width = sides[0]

        top_left, top_right, bottom_right, bottom_left = None, None, None, None
        x_cord = sorted(box[:, 0])
        y_cord = sorted(box[:, 1])
        for ele in box:
            points = [0, 0, 0, 0]  ## [left, right, top, bottom]
            if ele[0] in x_cord[:2]:
                points[0] = 1
            else:
                points[1] = 1
            if ele[1] in y_cord[:2]:
                points[2] = 1
            else:
                points[3] = 1
            if points[0] and points[3]:
                bottom_left = ele
            elif points[0] and points[2]:
                top_left = ele
            elif points[1] and points[2]:
                top_right = ele
            else:
                bottom_right = ele

        if top_left is not None and top_right is not None and bottom_right is not None and bottom_left is not None:
            dst = np.float32([[0, 0], [new_width, 0], [new_width, new_height], [0, new_height]])
            pts = np.float32([top_left, top_right, bottom_right, bottom_left])
            transform_mat = cv2.getPerspectiveTransform(pts, dst)
            new_img = cv2.warpPerspective(img, transform_mat, (new_width, new_height))
            new_img = cv2.resize(new_img, (256, 256))
            if show is not None:
                cv2.imshow(show, new_img)
            return new_img

    def _random_cropping(self, img):
        dx = random.randint(0, 32)
        dy = random.randint(0, 32)
        cropped_img = img[dx:256 - (32 - dx), dy:256 - (32 - dy)]
        return cropped_img

    def _extract_eyes(self, img: np.ndarray, index: int = None):
        init_width, init_height = img.shape[1], img.shape[0]
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.face_mesh.process(img_)
        if results.multi_face_landmarks:
            for _ in results.multi_face_landmarks:
                landmarks_draw = np.array(
                    [np.multiply([p.x, p.y], [init_width, init_height]).astype(int) for p in
                     results.multi_face_landmarks[0].landmark])
        else:
            return False

        rect_right_eye = cv2.minAreaRect(landmarks_draw[self.right_eye_cords, :])
        box_right_eye = cv2.boxPoints(rect_right_eye)
        rect_left_eye = cv2.minAreaRect(landmarks_draw[self.left_eye_cords, :])
        box_left_eye = cv2.boxPoints(rect_left_eye)

        left_eye_img = self._crop_wrap(img, box_left_eye)
        right_eye_img = self._crop_wrap(img, box_right_eye)

        if self.YCrCb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        if isinstance(right_eye_img, type(None)) or isinstance(left_eye_img, type(None)):
            return False

        img = self._random_cropping(img).astype(np.uint8)
        left_eye_img = self._random_cropping(left_eye_img).astype(np.uint8)
        right_eye_img = self._random_cropping(right_eye_img).astype(np.uint8)

        self.left_eye.append(left_eye_img)
        self.right_eye.append(right_eye_img)
        self.face.append(img)
        if index is not None:
            self.data_index.append(index)
        return True
        # return left_eye_img, right_eye_img, img

    def _data_generator(self):
        correct_labels = self._labels_init[self.correct_idx]
        correct_binary_face = self._binary_face_init[self.correct_idx]

        for left, right, face, binary_face, labels in zip(self.left_eye, self.right_eye, self.face, correct_binary_face,
                                                          correct_labels):
            if self.mirror:
                imgs = [[left, np.flip(left, axis=1)], [right, np.flip(right, axis=1)], [face, np.flip(face, axis=1)],
                        [binary_face, np.flip(binary_face, axis=1)], [labels, abs(np.subtract(labels, [1, 0]))]]
            else:
                imgs = [[left], [right], [face], [binary_face], [labels]]

            for left_img, right_img, img, bin_img, label in zip(imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]):
                yield (left_img / 255., right_img / 255., img / 255., bin_img), label

    def create_dataset(self, shuffle: bool = True, mirror: bool = False, YCrCb: bool = False):

        self.mirror = mirror
        self.YCrCb = YCrCb

        try:
            self._face_init = np.load('face.npy', mmap_mode='r')
            self._binary_face_init = np.load('binary_face.npy', mmap_mode='r')
            self._labels_init = np.load('labels.npy', mmap_mode='r')
            self.init_size = self._face_init.shape[0]
            print(f"face shape: {self._face_init.shape}\nbinary_face shape: {self._binary_face_init.shape}\n"
                  f"labels shape: {self._labels_init.shape}")
        except Exception as e:
            print('Could not load data! Error:\n', e)
            return 0
        time.sleep(1)

        if shuffle:
            indices = np.arange(self._face_init.shape[0])
            np.random.shuffle(indices)

            self._face_init = self._face_init[indices]
            self._binary_face_init = self._binary_face_init[indices]
            self._labels_init = self._labels_init[indices]

        for idx, face_image in enumerate(self._face_init):
            c = self._extract_eyes(img=face_image)
            if c: self.correct_idx.append(idx)

            if idx % 10 == 0:
                sys.stderr.write(
                    f"\r{idx}/{self.init_size} [" + "-" * int(idx / self.init_size * 50) + '>' + '_' * int(
                        (self.init_size - idx) / self.init_size * 50) + "]")
                sys.stderr.flush()

        self.size = len(self.face)
        if mirror: self.size *= 2

        dataset = tf.data.Dataset.from_generator(
            self._data_generator,
            output_signature=((
                                  tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name='Left_eye_image'),
                                  tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name='Right_eye_image'),
                                  tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name='face_image'),
                                  tf.TensorSpec(shape=(25, 25, 1), dtype=tf.uint8, name='binary_image')),
                              tf.TensorSpec(shape=(2,), dtype=tf.float32, name='output')
            )
        )


        return dataset, self.size

