"""Strategy Manager for NeuralRoulette-AI
Handles loading and running different betting strategies
"""

import importlib
import asyncio
import logging
from pathlib import Path

class StrategyManager:
    """Manages and runs advanced roulette betting strategies"""

    def __init__(self, strategy_name, balance=10.0, auto_train=False):
        self.strategy_name = strategy_name
        self.balance = balance
        self.auto_train = auto_train
        self.strategy = None
        self.last_prediction = None
        self.logger = logging.getLogger("strategy_manager")

    async def load_strategy(self):
        """Dynamically load the selected strategy"""
        try:
            class_name = f"{self.strategy_name.capitalize()}Strategy"
            module_name = f"src.strategies.{self.strategy_name}_strategy"
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            self.strategy = strategy_class(balance=self.balance, auto_train=self.auto_train)
            return True
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to load strategy '{self.strategy_name}': {str(e)}")
            return False

    async def process_number(self, number):
        """Process a new roulette number"""
        if not self.strategy:
            self.logger.error("No strategy loaded")
            return

        # Avalia a predição anterior (antes de adicionar o novo número)
        if self.last_prediction:
            predicted_zone = set()
            for num in self.last_prediction:
                predicted_zone.update(self.strategy.get_neighbors(num, n=3))

            bets = self.strategy.calculate_bets(self.last_prediction)
            total_bet = sum(bets.values())

            if number in predicted_zone:
                self.strategy.correct_predictions += 1
                winnings = bets[self.last_prediction[0]] * self.strategy.payout
                self.strategy.balance += winnings - total_bet
                win_status = f"✨ WIN! +${(winnings - total_bet):.2f}"
            else:
                self.strategy.balance -= total_bet
                win_status = f"❌ LOSS -${total_bet:.2f}"

            self.strategy.total_spins += 1
            win_rate = (self.strategy.correct_predictions / self.strategy.total_spins) * 100
            roi = ((self.strategy.balance - self.balance) / self.balance) * 100

            self.logger.info(
                f"Number: {number}, Balance: ${self.strategy.balance:.2f}, Win Rate: {win_rate:.2f}%, ROI: {roi:.2f}%"
            )

            print(f"\n🎲 Spin #{self.strategy.total_spins}")
            print(f"🎯 Result: {number} ({self.strategy.get_color(number)})")
            print(f"🔮 Predicted: {', '.join(str(n) for n in self.last_prediction)}")
            print("💰 Bets placed:")
            for bet_number, amount in bets.items():
                print(f"   {bet_number}: ${amount:.2f}")
            print(f"💵 Total bet: ${total_bet:.2f}")
            print(win_status)
            print(f"🏆 Win Rate: {win_rate:.2f}% ({self.strategy.correct_predictions}/{self.strategy.total_spins})")
            print(f"📈 ROI: {roi:.2f}%")
            print(f"💸 Balance: ${self.strategy.balance:.2f}")

        # Adiciona o número atual ao histórico
        self.strategy.game_history.append(number)

        # Gera nova predição para a próxima rodada
        if len(self.strategy.game_history) >= self.strategy.sequence_length:
            self.last_prediction = self.strategy.predict_numbers(self.strategy.game_history)
            print(f"🔮 Nova predição gerada para próxima rodada: {self.last_prediction}")

            if self.auto_train and len(self.strategy.game_history) >= 20:
                try:
                    X, y = self.strategy.preprocess_data(self.strategy.game_history)
                    if X is not None and len(X) > 0:
                        model = self.strategy.load_model()
                        model.fit(X, y, epochs=10, batch_size=min(64, len(X)), verbose=0)
                        model.save(self.strategy.model_file)
                        print("💾 Model updated with new data")
                except Exception as e:
                    print(f"⚠️ Training error: {str(e)}")
        else:
            print(f"📊 Not enough history yet ({len(self.strategy.game_history)}/{self.strategy.sequence_length}) - please wait, receiving numbers...")

        if len(self.strategy.game_history) > 1000:
            self.strategy.game_history = self.strategy.game_history[-1000:]

        if self.strategy.balance <= 0:
            print("❌ Balance depleted. Stopping strategy.")
            return False

        return True
