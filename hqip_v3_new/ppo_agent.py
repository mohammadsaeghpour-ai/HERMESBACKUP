"""
HERMES Trading Bot — Q-Learning Agent
یادگیری تعاملی واقعی بدون torch
"""
import numpy as np
import json
import os
from datetime import datetime
from collections import deque


class CryptoTradingEnv:
    """محیط ترید"""
    
    def __init__(self, data, lookback=20):
        self.data = data
        self.lookback = lookback
        self.reset()
    
    def reset(self):
        self.current_step = self.lookback
        self.position = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.wins = 0
        self.losses = 0
        self.portfolio_values = [10000]
        return self._get_state()
    
    def _get_state(self):
        """۱۰ ویژگی کلیدی"""
        if self.current_step >= len(self.data['close']):
            return np.zeros(10)
        
        c = self.data['close'][self.current_step]
        h = self.data['high'][self.current_step]
        l = self.data['low'][self.current_step]
        
        # RSI ساده
        prices = self.data['close'][max(0, self.current_step - 14):self.current_step + 1]
        if len(prices) > 1:
            deltas = np.diff(prices)
            gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
            losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-8
            rsi = 100 - (100 / (1 + gains / max(losses, 1e-8)))
        else:
            rsi = 50
        
        # EMA cross
        ema8 = np.mean(self.data['close'][self.current_step - 8:self.current_step + 1])
        ema20 = np.mean(self.data['close'][self.current_step - 20:self.current_step + 1]) if self.current_step >= 20 else ema8
        
        # Momentum
        momentum = (c - self.data['close'][self.current_step - 5]) / max(c, 1e-8) * 100
        
        # Volatility
        recent = self.data['close'][self.current_step - 10:self.current_step + 1]
        volatility = np.std(np.diff(recent) / recent[:-1]) * 100 if len(recent) > 1 else 0
        
        state = np.array([
            rsi / 100,                    # 0: RSI
            (ema8 / max(ema20, 1) - 1),   # 1: EMA cross
            momentum / 10,                 # 2: Momentum
            volatility * 10,               # 3: Volatility
            (h - c) / max(c, 1e-8) * 100, # 4: Wick up
            (c - l) / max(c, 1e-8) * 100, # 5: Wick down
            self.position,                 # 6: Current position
            self.total_pnl / 10,           # 7: PnL
            (c - self.data['close'][self.current_step - 1]) / max(c, 1e-8) * 100,  # 8: Price change
            self.wins / max(self.wins + self.losses, 1)  # 9: Win rate
        ], dtype=np.float32)
        
        return np.clip(state, -1, 1)
    
    def _discretize_state(self, state):
        """تبدیل state پیوسته به گسسته"""
        # هر ویژگی رو به ۵ بخش تقسیم می‌کنیم
        bins = 5
        discrete = []
        for i in range(len(state)):
            val = state[i]
            # -1 to 1 → 0 to bins-1
            bin_idx = int(np.clip((val + 1) / 2 * bins, 0, bins - 1))
            discrete.append(bin_idx)
        return tuple(discrete)
    
    def step(self, action):
        close = self.data['close'][self.current_step]
        next_close = self.data['close'][min(self.current_step + 1, len(self.data) - 1)]
        price_change = (next_close - close) / max(close, 1e-8)
        
        reward = 0
        
        if action == 1:  # BUY
            if self.position == -1:
                pnl = (self.entry_price - close) / self.entry_price * 20
                self.total_pnl += pnl
                reward = pnl * 5
                if pnl > 0: self.wins += 1
                else: self.losses += 1
                self.position = 0
            elif self.position == 0:
                self.position = 1
                self.entry_price = close
                reward = 0.3 + price_change * 10
            else:
                reward = price_change * 5
                
        elif action == 2:  # SELL
            if self.position == 1:
                pnl = (close - self.entry_price) / self.entry_price * 20
                self.total_pnl += pnl
                reward = pnl * 5
                if pnl > 0: self.wins += 1
                else: self.losses += 1
                self.position = 0
            elif self.position == 0:
                self.position = -1
                self.entry_price = close
                reward = 0.3 - price_change * 10
            else:
                reward = -price_change * 5
        else:  # HOLD
            if self.position == 1:
                reward = price_change * 3
            elif self.position == -1:
                reward = -price_change * 3
            else:
                reward = -0.02
        
        self.portfolio_values.append(10000 + self.total_pnl * 100)
        self.current_step += 1
        done = self.current_step >= len(self.data['close']) - 1
        
        next_state = self._get_state()
        return next_state, reward, done, {
            'pnl': self.total_pnl,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.wins / max(self.wins + self.losses, 1)
        }


class QLearningAgent:
    """ایجنت یادگیری Q-Learning"""
    
    def __init__(self, state_bins=10, n_actions=3):
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.lr = 0.15
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        # Q-Table — فقط 5 ویژگی کلیدی برای discretize
        self.key_features = [0, 1, 2, 3, 8]  # RSI, EMA, Mom, Vol, PriceChange
        self.q_table = {}
        self.memory = deque(maxlen=5000)
        self.update_count = 0
    
    def get_state_key(self, state):
        """discretize فقط ویژگی‌های کلیدی"""
        key = []
        for i in self.key_features:
            val = state[i] if i < len(state) else 0
            # map -1..1 to 0..bins-1
            bin_idx = int(np.clip((val + 1) / 2 * (self.state_bins - 1), 0, self.state_bins - 1))
            key.append(bin_idx)
        return tuple(key)
    
    def get_q(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        state_key = self.get_state_key(state)
        q_values = self.get_q(state_key)
        return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        current_q = self.get_q(state_key)
        next_q = self.get_q(next_state_key)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q)
        
        current_q[action] += self.lr * (target - current_q[action])
        self.update_count += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_stats(self):
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'updates': self.update_count,
            'memory_size': len(self.memory)
        }


class HERMESBot:
    """ربات معاملاتی HERMES"""
    
    def __init__(self):
        self.file_path = os.path.join(os.path.dirname(__file__), 'hermes_q_data.json')
    
    def create_data(self, n=2500):
        np.random.seed(42)
        price = 60000
        prices = [price]
        vol = 0.02
        for i in range(n - 1):
            vol = 0.95 * vol + 0.05 * abs(np.random.randn()) * 0.03
            price *= (1 + np.random.randn() * vol)
            prices.append(max(price, 1000))
        prices = np.array(prices)
        return {
            'open': prices * (1 + np.random.randn(n) * 0.005),
            'high': prices * (1 + abs(np.random.randn(n)) * 0.01),
            'low': prices * (1 - abs(np.random.randn(n)) * 0.01),
            'close': prices,
            'volume': np.random.exponential(1000, n)
        }
    
    def train(self, episodes=500):
        print(f"{'='*60}")
        print(f"  🤖 HERMES Bot — Q-Learning Training")
        print(f"{'='*60}")
        
        data = self.create_data(2500)
        train_data = {k: v[:2000] for k, v in data.items()}
        test_data = {k: v[2000:] for k, v in data.items()}
        
        env = CryptoTradingEnv(train_data)
        agent = QLearningAgent()
        
        rewards_history = []
        best_portfolio = 10000
        
        print(f"\n  📊 Episodes: {episodes}")
        
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            rewards_history.append(total_reward)
            portfolio = env.portfolio_values[-1]
            
            if portfolio > best_portfolio:
                best_portfolio = portfolio
            
            if (ep + 1) % 100 == 0:
                recent = rewards_history[-100:]
                avg_reward = np.mean(recent)
                print(f"\n  Episode {ep+1}/{episodes}")
                print(f"    Avg Reward: {avg_reward:.4f}")
                print(f"    Win Rate: {env.wins / max(env.wins + env.losses, 1):.2%}")
                print(f"    Portfolio: ${portfolio:,.2f}")
                print(f"    Best: ${best_portfolio:,.2f}")
                print(f"    Epsilon: {agent.epsilon:.4f}")
                print(f"    Q-Table: {len(agent.q_table)} states")
        
        # بک‌تست
        print(f"\n{'='*60}")
        print(f"  📈 بک‌تست")
        print(f"{'='*60}")
        
        agent.epsilon = 0  # بدون exploration
        test_env = CryptoTradingEnv(test_data)
        state = test_env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            state, _, done, _ = test_env.step(action)
        
        test_return = ((test_env.portfolio_values[-1] / test_env.portfolio_values[0]) - 1) * 100
        test_wr = test_env.wins / max(test_env.wins + test_env.losses, 1)
        
        print(f"    Return: {test_return:+.2f}%")
        print(f"    Win Rate: {test_wr:.2%}")
        print(f"    Trades: {test_env.wins + test_env.losses}")
        print(f"    P&L: ${test_env.total_pnl * 100:,.2f}")
        
        # ذخیره
        self.save(agent)
        
        return agent
    
    def predict(self, agent, market_data):
        env = CryptoTradingEnv(market_data)
        state = env._get_state()
        action = agent.choose_action(state)
        actions = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
        state_key = agent.get_state_key(state)
        q_values = agent.get_q(state_key)
        
        return {
            'action': actions[action],
            'action_id': action,
            'q_values': q_values.tolist(),
            'confidence': float(np.max(q_values))
        }
    
    def save(self, agent):
        data = {
            'q_table': {str(k): v.tolist() for k, v in agent.q_table.items()},
            'epsilon': agent.epsilon,
            'update_count': agent.update_count,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f)
        print(f"\n  ✅ ذخیره شد: {self.file_path}")
    
    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            agent = QLearningAgent()
            agent.q_table = {eval(k): np.array(v) for k, v in data['q_table'].items()}
            agent.epsilon = data.get('epsilon', 0.05)
            print(f"  ✅ بارگذاری شد")
            return agent
        return None


if __name__ == '__main__':
    bot = HERMESBot()
    agent = bot.train(episodes=500)
    
    print(f"\n{'='*60}")
    print(f"  🔮 پیش‌بینی")
    print(f"{'='*60}")
    
    test_data = bot.create_data(100)
    prediction = bot.predict(agent, test_data)
    print(f"    Action: {prediction['action']}")
    print(f"    Q-Values: {prediction['q_values']}")
    print(f"    Confidence: {prediction['confidence']:.4f}")
    
    stats = agent.get_stats()
    print(f"\n  📊 آمار:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
