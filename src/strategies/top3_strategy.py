import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Flatten, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, LSTM, Concatenate
from tensorflow.keras.utils import to_categorical
from collections import Counter
import os
import random

class Top3Strategy:
    """Top-3 Ultra Strategy: Transformer + LSTM + Monte Carlo + Walk-Forward"""

    def __init__(self, balance=10.0, auto_train=False):
        self.name = "Top-3 Ultra"
        self.description = "Transformer + LSTM + Cluster + Monte Carlo + Walk-Forward"
        self.balance = balance
        self.auto_train = auto_train
        self.game_history = []
        self.total_spins = 0
        self.correct_predictions = 0

        self.sequence_length = 10
        self.roulette_range = 37
        self.model_file = "models/top3_ultra_model.keras"
        self.epochs = 20
        self.batch_size = 32

        self.base_bet = 0.01
        self.max_bet_fraction = 0.05
        self.payout = 35

        self.red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
        self.black_numbers = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}

        self.roulette_wheel = [
            0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
            5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
        ]

    def get_color(self, number):
        if number == 0:
            return "green"
        return "red" if number in self.red_numbers else "black"

    def get_neighbors(self, number, n=3):
        if number not in self.roulette_wheel:
            return [number]
        idx = self.roulette_wheel.index(number)
        total = len(self.roulette_wheel)
        return [self.roulette_wheel[(idx + i) % total] for i in range(-n, n + 1)]

    def load_model(self):
        if os.path.exists(self.model_file):
            print(f"✅ Loading model: {self.model_file}")
            return load_model(self.model_file)
        else:
            print("🔄 Creating Transformer + LSTM model...")
            return self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.sequence_length,))
        x = Embedding(input_dim=self.roulette_range, output_dim=16)(input_layer)

        attn_output = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
        attn_output = LayerNormalization()(attn_output)

        lstm_output = LSTM(64, return_sequences=True)(x)
        lstm_output = LayerNormalization()(lstm_output)

        combined = Concatenate()([attn_output, lstm_output])
        x = Flatten()(combined)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        output = Dense(self.roulette_range, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def preprocess_data(self, data):
        X, y = [], []
        for i in range(0, len(data) - self.sequence_length, self.sequence_length):  # walk-forward
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        if not X:
            return None, None
        X = np.array(X)
        y = to_categorical(y, num_classes=self.roulette_range)
        return X, y

    def predict_numbers(self, recent_results):
        if len(recent_results) < self.sequence_length:
            return list(range(3))

        model = self.load_model()
        sequence = np.array(recent_results[-self.sequence_length:])
        sequence = sequence.reshape((1, self.sequence_length))
        probabilities = model.predict(sequence, verbose=0)[0]

        probabilities += self.get_hot_numbers_boost()
        top_indices = np.argsort(probabilities)[-3:][::-1]
        return [int(idx) for idx in top_indices]

    def get_hot_numbers_boost(self):
        if len(self.game_history) < 50:
            return np.zeros(self.roulette_range)
        counts = Counter(self.game_history[-100:])
        boost = np.zeros(self.roulette_range)
        for num, freq in counts.items():
            boost[num] = freq / 1000
        return boost

    def calculate_bets(self, predicted_numbers):
        roi = ((self.balance - 10.0) / 10.0)
        dynamic_bet = self.base_bet * (1 + roi)
        total_bet = min(self.balance * self.max_bet_fraction, len(predicted_numbers) * dynamic_bet)
        bet_per_number = total_bet / len(predicted_numbers)
        return {num: bet_per_number for num in predicted_numbers}

    def monte_carlo_simulation(self, spins=1000):
        results = []
        for _ in range(spins):
            simulated = random.choices(range(self.roulette_range), k=self.sequence_length)
            prediction = self.predict_numbers(simulated)
            result = random.choice(range(self.roulette_range))
            win = result in prediction
            results.append(win)
        win_rate = sum(results) / len(results)
        print(f"🎲 Monte Carlo Simulation: {win_rate:.2%} win rate over {spins} spins")

    def train_model(self):
        if len(self.game_history) < self.sequence_length + 10:
            return
        model = self.load_model()
        X, y = self.preprocess_data(self.game_history)
        if X is None:
            return
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        model.save(self.model_file)
        print("💾 Model updated with new data")
