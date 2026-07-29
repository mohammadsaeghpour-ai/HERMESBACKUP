"""
HQIP v3 — Interactive Learning System
Deep Learning + ML + Game Theory + Memory + Attention
"""
import numpy as np
import json
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional
import os

class MemoryBank:
    """حافظه بلندمدت بازار"""
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.short_term = deque(maxlen=100)
        self.long_term = deque(maxlen=max_size)
        self.patterns = {}
        self.rewards = []
        self.file_path = os.path.join(os.path.dirname(__file__), 'memory_data.json')
        self.load()
    
    def store(self, state, action, reward, next_state):
        """ذخیره تجربه"""
        experience = {
            'state': state.tolist() if hasattr(state, 'tolist') else list(state),
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist() if hasattr(next_state, 'tolist') else list(next_state),
            'timestamp': datetime.now().isoformat()
        }
        self.short_term.append(experience)
        self.long_term.append(experience)
        self.rewards.append(reward)
        self.save()
    
    def get_similar(self, state, k=5):
        """یافتن تجربه‌های مشابه"""
        if len(self.short_term) < k:
            return []
        
        state_arr = np.array(state) if not isinstance(state, np.ndarray) else state
        scores = []
        
        for exp in self.short_term:
            exp_state = np.array(exp['state'])
            if len(exp_state) == len(state_arr):
                dist = np.linalg.norm(state_arr - exp_state)
                scores.append((dist, exp))
        
        scores.sort(key=lambda x: x[0])
        return [s[1] for s in scores[:k]]
    
    def get_win_rate(self, window=100):
        """نرخ برد اخیر"""
        if len(self.rewards) < 10:
            return 0.5
        recent = self.rewards[-window:]
        wins = sum(1 for r in recent if r > 0)
        return wins / len(recent)
    
    def save(self):
        """ذخیره در فایل"""
        try:
            data = {
                'short_term': list(self.short_term)[-100:],
                'long_term': list(self.long_term)[-1000:],
                'rewards': self.rewards[-1000:],
                'patterns': self.patterns
            }
            with open(self.file_path, 'w') as f:
                json.dump(data, f)
        except:
            pass
    
    def load(self):
        """بارگذاری از فایل"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                self.short_term = deque(data.get('short_term', []), maxlen=100)
                self.long_term = deque(data.get('long_term', []), maxlen=self.max_size)
                self.rewards = data.get('rewards', [])
                self.patterns = data.get('patterns', {})
        except:
            pass


class AttentionMechanism:
    """مکانیزم توجه — تمرکز روی مهم‌ترین ویژگی‌ها"""
    def __init__(self, input_size, attention_size=32):
        self.input_size = input_size
        self.attention_size = attention_size
        self.W_query = np.random.randn(input_size, attention_size) * 0.01
        self.W_key = np.random.randn(input_size, attention_size) * 0.01
        self.W_value = np.random.randn(input_size, attention_size) * 0.01
        self.W_output = np.random.randn(attention_size, input_size) * 0.01
        self.learning_rate = 0.001
    
    def forward(self, x):
        """محاسبه توجه"""
        Q = x @ self.W_query
        K = x @ self.W_key
        V = x @ self.W_value
        
        attention_scores = Q @ K.T / np.sqrt(K.shape[1])
        attention_weights = self.softmax(attention_scores)
        
        attended = attention_weights @ V
        output = attended @ self.W_output
        return output, attention_weights
    
    def softmax(self, x):
        """نرم‌افزار softmax"""
        if len(x.shape) == 1:
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
    
    def update(self, x, target, gradients):
        """به‌روزرسانی وزن‌ها"""
        self.W_query -= self.learning_rate * gradients.get('W_query', 0)
        self.W_key -= self.learning_rate * gradients.get('W_key', 0)
        self.W_value -= self.learning_rate * gradients.get('W_value', 0)


class DeepNeuralNetwork:
    """شبکه عصبی عمیق"""
    def __init__(self, layers=[64, 32, 16, 1]):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.learning_rate = 0.001
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """پیش‌روی در شبکه"""
        self.activations = [x]
        
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            self.activations.append(a)
        
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        output = z
        self.activations.append(output)
        
        return output
    
    def backward(self, target, output):
        """عقبگرد و به‌روزرسانی"""
        m = self.activations[0].shape[0]
        delta = output - target
        
        for i in range(len(self.weights) - 1, -1, -1):
            dw = self.activations[i].T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_derivative(self.activations[i])
            
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db


class GameTheoryAgent:
    """ایجنت تئوری بازی"""
    def __init__(self):
        self.strategies = {
            'aggressive': {'win_weight': 1.2, 'loss_weight': 0.8},
            'conservative': {'win_weight': 0.8, 'loss_weight': 1.2},
            'balanced': {'win_weight': 1.0, 'loss_weight': 1.0}
        }
        self.current_strategy = 'balanced'
        self.history = []
    
    def decide_strategy(self, market_state, memory_win_rate):
        """تصمیم‌گیری استراتژی"""
        if memory_win_rate > 0.7:
            self.current_strategy = 'aggressive'
        elif memory_win_rate < 0.4:
            self.current_strategy = 'conservative'
        else:
            self.current_strategy = 'balanced'
        
        return self.current_strategy
    
    def calculate_payoff(self, action, outcome):
        """محاسبه بازده"""
        weights = self.strategies[self.current_strategy]
        if outcome > 0:
            return outcome * weights['win_weight']
        else:
            return outcome * weights['loss_weight']
    
    def nash_equilibrium(self, opponent_actions):
        """محاسبه تعادل نش"""
        if not opponent_actions:
            return 'balanced'
        
        avg_opponent = np.mean(opponent_actions)
        
        if avg_opponent > 0.6:
            return 'conservative'
        elif avg_opponent < 0.4:
            return 'aggressive'
        else:
            return 'balanced'


class ReinforcementLearning:
    """یادگیری تقویتی"""
    def __init__(self, state_size=10, action_size=3, learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table = {}
    
    def get_state_key(self, state):
        """تبدیل state به کلید"""
        if isinstance(state, np.ndarray):
            state = state.tolist()
        return json.dumps([round(x, 2) for x in state[:self.state_size]])
    
    def get_q_value(self, state, action):
        """دریافت مقدار Q"""
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        return self.q_table[key][action]
    
    def choose_action(self, state):
        """انتخاب عمل"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[key])
    
    def update(self, state, action, reward, next_state):
        """به‌روزرسانی Q-Table"""
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)
        
        current_q = self.q_table[key][action]
        max_next_q = np.max(self.q_table[next_key])
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[key][action] = new_q


class PatternRecognizer:
    """تشخیص الگوها"""
    def __init__(self):
        self.patterns = {
            'double_bottom': {'reliability': 0.75, 'direction': 'buy'},
            'double_top': {'reliability': 0.75, 'direction': 'sell'},
            'head_shoulders': {'reliability': 0.80, 'direction': 'sell'},
            'inverse_head_shoulders': {'reliability': 0.80, 'direction': 'buy'},
            'ascending_triangle': {'reliability': 0.65, 'direction': 'buy'},
            'descending_triangle': {'reliability': 0.65, 'direction': 'sell'},
            'bull_flag': {'reliability': 0.70, 'direction': 'buy'},
            'bear_flag': {'reliability': 0.70, 'direction': 'sell'},
        }
    
    def detect(self, prices):
        """تشخیص الگو"""
        if len(prices) < 20:
            return None
        
        prices = np.array(prices)
        
        # Double Bottom
        min_idx = np.argmin(prices[-20:])
        if min_idx > 5 and min_idx < 15:
            left_min = np.min(prices[-20:-10])
            right_min = np.min(prices[-10:])
            if abs(left_min - right_min) / left_min < 0.02:
                return self.patterns['double_bottom']
        
        # Double Top
        max_idx = np.argmax(prices[-20:])
        if max_idx > 5 and max_idx < 15:
            left_max = np.max(prices[-20:-10])
            right_max = np.max(prices[-10:])
            if abs(left_max - right_max) / left_max < 0.02:
                return self.patterns['double_top']
        
        return None


class InteractiveLearningSystem:
    """سیستم یادگیری تعاملی"""
    def __init__(self):
        self.memory = MemoryBank()
        self.attention = AttentionMechanism(input_size=10)
        self.dnn = DeepNeuralNetwork([10, 32, 16, 8, 1])
        self.game_theory = GameTheoryAgent()
        self.rl = ReinforcementLearning(state_size=10, action_size=3)
        self.pattern_recognizer = PatternRecognizer()
        
        self.trade_count = 0
        self.total_pnl = 0
        self.history = []
    
    def extract_features(self, prices, indicators):
        """استخراج ویژگی‌ها"""
        features = []
        
        # قیمت‌ها
        if len(prices) >= 20:
            features.extend([
                (prices[-1] - prices[-20]) / prices[-20],  # تغییر ۲۰ کندلی
                (prices[-1] - prices[-10]) / prices[-10],  # تغییر ۱۰ کندلی
                (prices[-1] - prices[-5]) / prices[-5],    # تغییر ۵ کندلی
                np.std(prices[-20:]) / np.mean(prices[-20:]),  # نوسان
            ])
        
        # اندیکاتورها
        for key in ['rsi', 'adx', 'mom', 'trend', 'volume']:
            if key in indicators:
                features.append(indicators[key])
        
        # پر کردن
        while len(features) < 10:
            features.append(0)
        
        return np.array(features[:10])
    
    def predict(self, features):
        """پیش‌بینی"""
        # توجه
        attended, weights = self.attention.forward(features.reshape(1, -1))
        
        # شبکه عصبی
        prediction = self.dnn.forward(attended)
        
        # تئوری بازی
        strategy = self.game_theory.decide_strategy(
            features, self.memory.get_win_rate()
        )
        
        # RL
        rl_action = self.rl.choose_action(features)
        
        return {
            'prediction': float(prediction[0][0]),
            'attention_weights': weights.tolist(),
            'strategy': strategy,
            'rl_action': rl_action,
            'confidence': self.memory.get_win_rate()
        }
    
    def learn(self, state, action, reward, next_state):
        """یادگیری"""
        # ذخیره در حافظه
        self.memory.store(state, action, reward, next_state)
        
        # به‌روزرسانی RL
        self.rl.update(state, action, reward, next_state)
        
        # به‌روزرسانی DNN
        features = state.reshape(1, -1) if isinstance(state, np.ndarray) else np.array(state).reshape(1, -1)
        output = self.dnn.forward(features)
        target = np.array([[reward]])
        self.dnn.backward(target, output)
        
        # به‌روزرسانی آمار
        self.trade_count += 1
        self.total_pnl += reward
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'reward': reward,
            'total_pnl': self.total_pnl,
            'win_rate': self.memory.get_win_rate()
        })
    
    def get_stats(self):
        """آمار سیستم"""
        return {
            'trade_count': self.trade_count,
            'total_pnl': self.total_pnl,
            'win_rate': self.memory.get_win_rate(),
            'avg_reward': self.total_pnl / max(1, self.trade_count),
            'memory_size': len(self.memory.short_term),
            'strategy': self.game_theory.current_strategy
        }
    
    def save(self):
        """ذخیره سیستم"""
        self.memory.save()
    
    def load(self):
        """بارگذاری سیستم"""
        self.memory.load()
