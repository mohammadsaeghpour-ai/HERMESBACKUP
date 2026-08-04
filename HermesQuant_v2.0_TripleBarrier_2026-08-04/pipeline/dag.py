"""Directed Acyclic Graph for Agent Dependencies"""
from collections import defaultdict
from enum import Enum

class Stage(Enum):
    DATA = 0
    INDEPENDENT = 1
    META = 2
    STRUCTURE = 3
    DECISION = 4
    RISK = 5

class AgentDAG:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.adj = defaultdict(list)
        self._build()
    
    def _build(self):
        ind = {'Trend': 1.5, 'Momentum': 1.3, 'Volume': 1.3,
               'Volatility': 1.0, 'Pattern': 1.1, 'DLForecast': 1.0}
        for n, w in ind.items():
            self.add(n, Stage.INDEPENDENT, w)
        
        self.add('Regime', Stage.META, 1.0, ['Volatility', 'Volume', 'Trend'])
        self.add('MarketStructure', Stage.META, 1.4, ['Trend'])
        self.add('Whale', Stage.META, 1.5, ['Volume'])
        
        self.add('Liquidity', Stage.STRUCTURE, 1.5, ['MarketStructure'])
        self.add('SMC', Stage.STRUCTURE, 1.0, ['Liquidity', 'MarketStructure'])
        self.add('Wyckoff', Stage.STRUCTURE, 1.4, ['Volume', 'Trend', 'MarketStructure'])
        self.add('MathBrain', Stage.STRUCTURE, 1.4, ['Trend', 'Momentum', 'Volatility'])
        
        self.add('GameTheory', Stage.DECISION, 1.3, ['Regime', 'Whale'])
        self.add('SmartAction', Stage.DECISION, 1.7, ['Trend', 'Momentum', 'SMC', 'Regime'])
        self.add('ML', Stage.DECISION, 1.2, ['Trend', 'Momentum', 'Volume', 'Volatility', 'Pattern'])
        
        self.add('Risk', Stage.RISK, 0.0, ['Regime', 'Volatility'])
    
    def add(self, name, stage, weight=1.0, deps=None):
        self.nodes[name] = {'stage': stage, 'weight': weight, 'deps': deps or []}
        for d in (deps or []):
            self.edges.append((d, name))
            self.adj[d].append(name)
    
    def topological_sort(self):
        layers = defaultdict(list)
        for n, info in self.nodes.items():
            layers[info['stage'].value].append(n)
        return [layers[i] for i in sorted(layers.keys())]
    
    def print(self):
        for i, layer in enumerate(self.topological_sort()):
            for n in layer:
                info = self.nodes[n]
                deps = ", ".join(info['deps']) or "none"
                print("  STAGE %d: %s [w=%.1f] <- %s" % (i, n, info['weight'], deps))
