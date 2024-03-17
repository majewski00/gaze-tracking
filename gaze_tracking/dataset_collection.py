import os
import sys
import random
from copy import copy
from os.path import join, exists
from time import perf_counter
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pygame as py


class GazeCollector:
    """
    A class for collecting facial image datasets using Pygame and OpenCV.
    """

    def __init__(self):
        ## PyGame
        py.init()
        self.py_clock = py.time.Clock()
        self.screen = None
        self.show_buttons = False
        self.close_n = py.transform.scale(py.image.load(join('icons', 'close.png')), (30, 30))
        self.close_p = py.transform.scale(py.image.load(join('icons', 'close_active.png')), (30, 30))
        self.closeRect = self.close_n.get_rect()
        self.closeRect.center = (1920 - 25, 25)

        self.reset_n = py.transform.scale(py.image.load(join('icons', 'reset.png')), (30, 30))
        self.reset_p = py.transform.scale(py.image.load(join('icons', 'reset_active.png')), (30, 30))
        self.resetRect = self.reset_n.get_rect()
        self.resetRect.center = (1920 - 65, 25)
        self.pygame_running = False
        self.face_state = [0, 0, 0, 0]  ## up, right, down, left
        self.close_screen, self.close_timer = False, 0
        self.start_time = None
        self.medium_font = py.font.SysFont('georgia', 20)
        self.small_font = py.font.SysFont('georgia', 15)

        ## Moving dot
        self.init_pos = [random.randint(3, 13), random.randint(6, 13)]  #
        self.pos = [self.init_pos[0], self.init_pos[1]]
        self.dot_timer = perf_counter()
        self.speed_x, self.speed_y = 5, 30
        self.move: bool = None
        self.photo_interval = random.randint(20, 35)

        self.working_directory = None
        self.img = None
        self.main_list, self.save_count, self.label_list, self.row_save_count = [], 0, [], 0
        self.horizontal = None

        ## cv2 and mediapipe
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)

    def collect(self, dataset_directory: str, show: Optional[bool] = False, horizontal: Optional[bool] = True):
        """
        Main function to start collecting facial image dataset using Pygame.

        Args:
            dataset_directory: Name or path to existing or new dataset directory
            show: Optional setting to display cv2 window with camera display
            horizontal: Optional setting to chose whether to collect data in horizontal or vertical way. Changing this option might be useful to reduce inaccuracies in the edges. Default mode is horizontal.

        """
        self.working_directory = dataset_directory
        self.horizontal = horizontal
        while 1:
            for event in py.event.get():
                if event.type == py.QUIT:
                    exit()
            self._camera(show=show)
            self._board()

            self.py_clock.tick(60)

    def _camera(self, show: bool = True):
        success, img = self.cap.read()
        if success:
            self.img = cv2.flip(img, 1)
            init_height, init_width = self.img.shape[:2]
            img_ = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            results = self.face_mesh.process(img_)
            if results.multi_face_landmarks:
                for _ in results.multi_face_landmarks:
                    landmarks_draw = np.array(
                        [np.multiply([p.x, p.y], [init_width, init_height]).astype(int) for p in
                         results.multi_face_landmarks[0].landmark])
            else:
                return

            if min(landmarks_draw[:, 1]) < 0:
                self.face_state[0] = 1
            else:
                self.face_state[0] = 0
            if min(landmarks_draw[:, 0]) < 0:
                self.face_state[3] = 1
            else:
                self.face_state[3] = 0
            if max(landmarks_draw[:, 1]) >= self.init_height:
                self.face_state[2] = 1
            else:
                self.face_state[2] = 0
            if max(landmarks_draw[:, 0]) >= self.init_width:
                self.face_state[1] = 1
            else:
                self.face_state[1] = 0

            if show:
                cv2.imshow("Smile", self.img)

    def _binary_face_grid(self, rect):
        """
        Creates 25x25 binary grid and assign True to cells representing face image in original image.
        """
        binary_grid = np.zeros((25, 25, 1)).astype(int)
        for y_step in range(25):
            for x_step in range(25):
                _rect = ((25 * (x_step + 1), 25 * (y_step + 1)), (25, 25), 0)
                intersection_info = cv2.rotatedRectangleIntersection(rect, _rect)
                if intersection_info[0] == 1:
                    binary_grid[y_step, x_step, :] = 1

        for y_step in range(25):
            for x_step in range(25):
                if np.count_nonzero(binary_grid[y_step, x_step:, 0]) > 0 and np.count_nonzero(
                        binary_grid[y_step, :x_step, 0]) > 0:
                    binary_grid[y_step, x_step, :] = 1

        return binary_grid

    def _crop_wrap(self, box, margin: int = 15):
        """
        This function will crop face image so it will contain face in normalized size of 256 x 256 px

        Args:
            box: cv2.boxPoints(rect_face)
            margin: The margin in pixels that will be added to the most external landmarks of the face

        Returns:
            Normalized face image
        """

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
            pts = np.float32([top_left - [margin, margin], top_right + [margin, -margin],
                              bottom_right + [margin, -margin], bottom_left - [margin, margin]])
            transform_mat = cv2.getPerspectiveTransform(pts, dst)
            new_img = cv2.warpPerspective(self.img, transform_mat, (new_width, new_height))
            new_img = cv2.resize(new_img, (256, 256))

            return new_img

        return None

    def _face_prep(self):
        binary_face = []
        face = []
        labels = []

        for idx in range(len(self.main_list)):
            img, label = self.main_list[idx], self.label_list[idx]

            if idx % 10 == 0:
                sys.stderr.write(
                    f"\r{idx}/{len(self.main_list)} [" + "-" * int(idx / len(self.main_list) * 50) + '>' + '_' * int(
                        (len(self.main_list) - idx) / len(self.main_list) * 50) + "]")
                sys.stderr.flush()

            init_width, init_height = img.shape[1], img.shape[0]
            img_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            results = self.face_mesh.process(img_)
            if results.multi_face_landmarks:
                for _ in results.multi_face_landmarks:
                    landmarks_draw = np.array(
                        [np.multiply([p.x, p.y], [init_width, init_height]).astype(int) for p in
                         results.multi_face_landmarks[0].landmark])
            else:
                continue

            ## Blink Detection
            left_blink, right_blink = False, False
            if np.sum(abs(landmarks_draw[145] - landmarks_draw[159])) < np.sum(
                    abs(landmarks_draw[469] - landmarks_draw[471])) / 3.5:
                left_blink = True
            if np.sum(abs(landmarks_draw[374] - landmarks_draw[386])) < np.sum(
                    abs(landmarks_draw[474] - landmarks_draw[476])) / 3.5:
                right_blink = True
            if left_blink or right_blink:
                continue

            rect_face = cv2.minAreaRect(landmarks_draw)
            box_face = cv2.boxPoints(rect_face)

            self.img = img
            face_img = self._crop_wrap(box_face, margin=15)
            binary_face_img = self._binary_face_grid(rect=rect_face)

            if isinstance(face_img, type(None)):
                continue

            labels.append(label)
            binary_face.append(binary_face_img)
            face.append(face_img)

        print("\nData Preparation Ended\nSaving files...")
        final_data = [np.array(labels, dtype=np.float16), np.array(binary_face, dtype=np.uint8),
                      np.array(face, dtype=np.float32)]

        names = ['labels', 'binary_face', 'face']

        for name, data in zip(names, final_data):
            if exists(join(self.working_directory, name + '.npy')):
                print('  Detected file: ', name)
                old_data = np.load(file=join(self.working_directory, name + '.npy'))
                print('  Old data size: ', old_data.shape)
                data = np.concatenate((data, old_data), axis=0)
                print('  New data shape: ', data.shape)
            else:
                print("  Didn't detect file: ", name)
                print('  New data shape: ', data.shape)
            print("* * " * 10)
            np.save(file=join(self.working_directory, name + '.npy'), arr=data)

    def _board(self, save: bool = True):
        """
        Logic for Pygame Window and moving dot.
        """
        if not self.pygame_running:
            self.pygame_running = True
            self.screen = py.display.set_mode((0, 0), py.FULLSCREEN)
            self.start_time = perf_counter()

        self.screen.fill((219, 205, 200))
        save_and_quit = False

        ## Error frame (no face / face partially invisible)
        if self.face_state[0] == 1:
            py.draw.line(self.screen, (163, 31, 60), (0, 0), (1920, 0), 9)
            if self.horizontal:
                self.pos[0] = self.init_pos[0]
            else:
                self.pos[1] = self.init_pos[1]
            self.move = None
            self.dot_timer = perf_counter()
            if self.row_save_count > 0 and not self.close_screen:
                for _ in range(self.row_save_count):
                    self.main_list.pop(len(self.main_list) - 1)
                self.row_save_count = 0
        if self.face_state[1] == 1:
            py.draw.line(self.screen, (163, 31, 60), (1915, 0), (1915, 1080), 9)
            if self.horizontal:
                self.pos[0] = self.init_pos[0]
            else:
                self.pos[1] = self.init_pos[1]
            self.move = None
            self.dot_timer = perf_counter()
            if self.row_save_count > 0 and not self.close_screen:
                for _ in range(self.row_save_count):
                    self.main_list.pop(len(self.main_list) - 1)
                self.row_save_count = 0
        if self.face_state[2] == 1:
            py.draw.line(self.screen, (163, 31, 60), (0, 1080), (1915, 1080), 9)
            if self.horizontal:
                self.pos[0] = self.init_pos[0]
            else:
                self.pos[1] = self.init_pos[1]
            self.move = None
            self.dot_timer = perf_counter()
            if self.row_save_count > 0 and not self.close_screen:
                for _ in range(self.row_save_count):
                    self.main_list.pop(len(self.main_list) - 1)
                self.row_save_count = 0
        if self.face_state[3] == 1:
            py.draw.line(self.screen, (163, 31, 60), (0, 0), (0, 1080), 9)
            if self.horizontal:
                self.pos[0] = self.init_pos[0]
            else:
                self.pos[1] = self.init_pos[1]
            self.move = None
            self.dot_timer = perf_counter()
            if self.row_save_count > 0 and not self.close_screen:
                for _ in range(self.row_save_count):
                    self.main_list.pop(len(self.main_list) - 1)
                self.row_save_count = 0

        ## Buttons
        mouse_pos = py.mouse.get_pos()
        if mouse_pos[0] > 1820 and mouse_pos[1] < 60:

            self.show_buttons = True
            self.move, self.dot_timer = None, perf_counter() - 2

            if not self.closeRect[0] < mouse_pos[0] < self.closeRect[0] + 30 or not 10 < mouse_pos[1] < 40:
                self.screen.blit(self.close_n, self.closeRect)

            else:
                self.screen.blit(self.close_p, self.closeRect)
                if py.mouse.get_pressed(3)[0]:
                    self.close_screen = True
                    self.close_timer = perf_counter()

            if not self.resetRect[0] < mouse_pos[0] < self.resetRect[0] + 30 or not 10 < mouse_pos[1] < 40:
                self.screen.blit(self.reset_n, self.resetRect)
            else:
                self.screen.blit(self.reset_p, self.resetRect)
                if py.mouse.get_pressed(3)[0]:
                    pass
        elif self.close_screen:
            self.show_buttons = True
        else:
            self.show_buttons = False

        ## Moving Dot
        if not self.close_screen:
            if self.move is None and perf_counter() - self.dot_timer > 4:
                self.move = True

            if self.move and perf_counter() - self.dot_timer > 2:
                if self.horizontal:
                    if self.pos[0] == self.init_pos[0]:
                        self.pos[0] += (self.speed_x // 2)
                        self.slow_timer = perf_counter()

                    if self.pos[0] < 80 or self.pos[0] > 1920 - 90:
                        self.pos[0] += (self.speed_x // 2)
                    else:
                        self.pos[0] += (self.speed_x // 1)

                    if self.pos[1] >= 1070 - self.speed_y and self.pos[0] >= 1920 - 20:
                        self.pos = [10, 10]
                        self.move = False
                        if save:
                            save_and_quit = True

                    if self.pos[0] > 1920 - 20:
                        self.dot_timer = perf_counter()
                        self.pos[0] = self.init_pos[0]
                        self.pos[1] += self.speed_y
                        self.row_save_count = 0

                else:
                    if self.pos[1] == self.init_pos[1]:
                        self.pos[1] += (self.speed_x // 2)
                        self.slow_timer = perf_counter()

                    if self.pos[1] < 80 or self.pos[1] > 1080 - 70:
                        self.pos[1] += (self.speed_x // 2)
                    else:
                        self.pos[1] += (self.speed_x // 1)

                    if self.pos[1] >= 1070 - self.speed_y and self.pos[0] >= 1920 - 20:
                        self.pos = [10, 10]
                        self.move = False
                        if save:
                            save_and_quit = True
                    if self.pos[1] > 1080 - 10:
                        self.dot_timer = perf_counter()
                        self.pos[1] = self.init_pos[1]
                        self.pos[0] += self.speed_y
                        self.row_save_count = 0
        else:
            py.draw.rect(self.screen, (182, 182, 182), (736, 416, 448, 248), 0)
            py.draw.rect(self.screen, (145, 141, 122), (740, 420, 440, 240), 0)

            if 740 < mouse_pos[0] < 960 and 600 < mouse_pos[1] < 660:
                py.draw.rect(self.screen, (158, 73, 90), (738, 601, 221, 59), 3)
                if py.mouse.get_pressed(3)[0]:
                    py.quit()
                    exit()
            if 960 < mouse_pos[0] < 1180 and 600 < mouse_pos[1] < 660:
                py.draw.rect(self.screen, (28, 102, 42), (958, 601, 221, 59), 3)
                if py.mouse.get_pressed(3)[0]:
                    save_and_quit = True
            if (736 > mouse_pos[0] or mouse_pos[0] > 1180) and (416 > mouse_pos[1] or mouse_pos[1] > 660):
                if py.mouse.get_pressed(3)[0]:
                    if perf_counter() - self.close_timer > 0.5:
                        self.close_timer = perf_counter()

                        self.close_screen = False
                        self.move = None
                        if self.row_save_count > 0:
                            self.save_count -= self.row_save_count
                            for _ in range(self.row_save_count):
                                self.main_list.pop(len(self.main_list) - 1)
                            self.row_save_count = 0
                        if self.horizontal:
                            self.pos[0] = 10
                        else:
                            self.pos[1] = 10

            ask_1 = self.medium_font.render('Do you want to save your progress ?', True, (28, 26, 26))
            ask_2 = self.small_font.render(f'The {str(self.save_count)} photos will be lost.', True, (56, 55, 55))

            self.screen.blit(ask_1, (740 + 50, 430))
            self.screen.blit(ask_2, (740 + 120, 420 + 50))

            py.draw.line(self.screen, (56, 55, 55), (740, 600), (1180, 600), 1)
            py.draw.line(self.screen, (56, 55, 55), (960, 600), (960, 659), 1)
            no = self.small_font.render('NO', True, (56, 55, 55))
            yes = self.small_font.render('YES', True, (56, 55, 55))
            self.screen.blit(no, (740 + 90, 600 + 25))
            self.screen.blit(yes, (960 + 90, 600 + 25))
            save = False

        py.draw.circle(self.screen, (224, 144, 187), self.pos, 8)
        py.draw.circle(self.screen, (38, 36, 36), self.pos, 2)

        if save_and_quit:
            py.quit()
            cv2.destroyAllWindows()
            self._face_prep()
            exit()

        ## Move dot in horizontal or vertical direction
        if self.horizontal:
            if 0 <= self.pos[0] % self.photo_interval <= 5 and save:
                self.save_count += 1
                self.row_save_count += 1
                self.main_list.append(self.img)
                c = copy(self.pos)
                self.label_list.append((c[0] / 1920, c[1] / 1080))
        else:
            if 0 <= self.pos[1] % self.photo_interval <= 5 and save:
                self.save_count += 1
                self.row_save_count += 1
                self.main_list.append(self.img)
                c = copy(self.pos)
                self.label_list.append((c[0] / 1920, c[1] / 1080))

        py.display.update()
