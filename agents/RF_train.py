# RF_train.py
import numpy as np
import sys
import os
import pickle

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from game.engine.dealer import Dealer
from random_player import RandomPlayer
from RF_player import RFPlayer

# 參數
num_episodes = 5000

# 資料集
obs_list = []
action_list = []

# 初始化遊戲環境
dealer = Dealer(small_blind_amount=20, initial_stack=1000)
rf_player = RFPlayer()
random_player = RandomPlayer()

dealer.register_player("rf_player", rf_player)
dealer.register_player("random_player", random_player)

print("Collecting data...")
for episode in range(num_episodes):
    dealer.set_verbose(0)
    dealer.start_game(max_round=1)
    
    # 收集RF玩家的觀察和動作
    obs_list.extend(rf_player.observation_history)
    action_list.extend(rf_player.action_history)
    
    if (episode+1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes} collected")

# 轉成 numpy
X = np.array(obs_list)
y = np.array(action_list)

# 分 train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練 RF
print("\nTraining Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate
y_pred = rf_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 儲存 model
with open("poker_policy_rf.pkl", "wb") as f:
    pickle.dump(rf_clf, f)
print("\nModel saved to poker_policy_rf.pkl")
