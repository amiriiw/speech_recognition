"""-------------------------------------------------------------------------
Welcome, this is amiriiw, this is a simple project about Speech recognition.
This file is the file where we run the game and use the model.
-----------------------------------------------------------"""
import queue  # https://docs.python.org/3/library/queue.html
import pygame  # https://www.pygame.org/docs/
import random  # https://docs.python.org/3/library/random.html
import threading  # https://docs.python.org/3/library/threading.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import tensorflow as tf  # https://www.tensorflow.org/
import sounddevice as sd  # https://python-sounddevice.readthedocs.io/en/0.4.7/
"""-------------------------------------------------------------------------"""


class SnakeGame:
    def __init__(self):
        pygame.init()
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.dis_width = 600
        self.dis_height = 400
        self.dis = pygame.display.set_mode((self.dis_width, self.dis_height))
        pygame.display.set_caption('Voice Controlled Snake Game')
        self.clock = pygame.time.Clock()
        self.snake_block = 10
        self.snake_speed = 5
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.score_font = pygame.font.SysFont("comicsansms", 35)
        self.x1_change = 0
        self.y1_change = 0

    def draw_snake(self, snake_block, snake_list):
        for x in snake_list:
            pygame.draw.rect(self.dis, self.white, [x[0], x[1], snake_block, snake_block])

    def display_score(self, score):
        value = self.score_font.render(f"Your Score: {score}", True, self.white)
        self.dis.blit(value, [0, 0])

    def display_message(self, msg, color):
        mesg = self.font_style.render(msg, True, color)
        self.dis.blit(mesg, [self.dis_width / 6, self.dis_height / 3])

    def game_loop(self):
        game_over = False
        game_close = False
        x1 = self.dis_width / 2
        y1 = self.dis_height / 2
        self.x1_change = 0
        self.y1_change = 0
        snake_list = []
        length_of_snake = 1
        foodx = round(random.randrange(0, self.dis_width - self.snake_block) / 10.0) * 10.0
        foody = round(random.randrange(0, self.dis_height - self.snake_block) / 10.0) * 10.0
        while not game_over:
            while game_close:
                self.dis.fill(self.black)
                self.display_message("You Lost! Press Q-Quit or C-Play Again", self.red)
                self.display_score(length_of_snake - 1)
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            game_over = True
                            game_close = False
                        if event.key == pygame.K_c:
                            self.game_loop()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
            if x1 >= self.dis_width or x1 < 0 or y1 >= self.dis_height or y1 < 0:
                game_close = True
            x1 += self.x1_change
            y1 += self.y1_change
            self.dis.fill(self.black)
            pygame.draw.circle(self.dis, self.red, [int(foodx), int(foody)], self.snake_block // 2)
            snake_head = [x1, y1]
            snake_list.append(snake_head)
            if len(snake_list) > length_of_snake:
                del snake_list[0]
            for x in snake_list[:-1]:
                if x == snake_head:
                    game_close = True
            self.draw_snake(self.snake_block, snake_list)
            self.display_score(length_of_snake - 1)
            pygame.display.update()
            if x1 == foodx and y1 == foody:
                foodx = round(random.randrange(0, self.dis_width - self.snake_block) / 10.0) * 10.0
                foody = round(random.randrange(0, self.dis_height - self.snake_block) / 10.0) * 10.0
                length_of_snake += 1
            self.clock.tick(self.snake_speed)
        pygame.quit()


class VoiceControl:
    label_names = ['down', 'left', 'right', 'up']
    model = tf.saved_model.load("saved")
    q = queue.Queue()

    @staticmethod
    def get_spectrogram(waveform):
        spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    @staticmethod
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        VoiceControl.q.put(indata.copy())

    @staticmethod
    def process_audio(confidence_threshold=0.85):
        while True:
            if not VoiceControl.q.empty():
                audio_data = VoiceControl.q.get()
                audio_data = np.squeeze(audio_data)
                if len(audio_data) == 16000:
                    audio_data = audio_data[tf.newaxis, ...]
                    result = VoiceControl.model(audio_data)
                    probabilities = tf.nn.softmax(result['predictions'][0])
                    max_prob = tf.reduce_max(probabilities)
                    if max_prob > confidence_threshold:
                        class_ids = tf.argmax(result['predictions'], axis=-1)
                        class_name = VoiceControl.label_names[class_ids.numpy()[0]]
                        print(f'Predicted class: {class_name}, Confidence: {max_prob:.2f}')
                        if class_name == 'left':
                            snake_game.x1_change = -snake_game.snake_block
                            snake_game.y1_change = 0
                        elif class_name == 'right':
                            snake_game.x1_change = snake_game.snake_block
                            snake_game.y1_change = 0
                        elif class_name == 'up':
                            snake_game.y1_change = -snake_game.snake_block
                            snake_game.x1_change = 0
                        elif class_name == 'down':
                            snake_game.y1_change = snake_game.snake_block
                            snake_game.x1_change = 0


VoiceControl.duration = 1
VoiceControl.fs = 16000
VoiceControl.stream = sd.InputStream(callback=VoiceControl.audio_callback, channels=1, samplerate=VoiceControl.fs, blocksize=int(VoiceControl.duration * VoiceControl.fs))
VoiceControl.stream.start()
VoiceControl.thread = threading.Thread(target=VoiceControl.process_audio, daemon=True)
VoiceControl.thread.start()
snake_game = SnakeGame()
snake_game.game_loop()
"""----------------"""
