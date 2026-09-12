import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class ReplayBuffer():
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)

    def add(self, s_img, s_vec, action, reward, ns_img, ns_vec, done):
        self.memory.append((s_img, s_vec, action, reward, ns_img, ns_vec, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        
        s_img, s_vec, a, r, ns_img, ns_vec, d = zip(*batch)
        
        return (np.stack(s_img), np.stack(s_vec), 
                np.array(a), np.array(r, dtype=np.float32), 
                np.stack(ns_img), np.stack(ns_vec), 
                np.array(d, dtype=np.float32))

class DroneNet(nn.Module):
    """Mạng DQN Tiêu chuẩn (Standard DQN) với 1 luồng q_stream trực tiếp."""
    def __init__(self, n_actions=5, state_vector_dim=23, img_shape=(3, 64, 64)):
        super(DroneNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_img = torch.zeros(1, *img_shape)
            conv_out_dim = self.conv(dummy_img).shape[1]
            
        self.img_fc = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        self.vector_fc = nn.Sequential(
            nn.Linear(state_vector_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        combined_dim = 128 + 128
        
        self.q_stream = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, image, state_vector):
        img_feat = self.img_fc(self.conv(image))
        vec_feat = self.vector_fc(state_vector)
        
        vec_feat = vec_feat.view(vec_feat.size(0), -1)
        combined = torch.cat((img_feat, vec_feat), dim=1)
        
        q_values = self.q_stream(combined)
        return q_values

class DQN():
    """Thuật toán Standard DQN."""
    def __init__(self, model, n_actions, memory_size=50000, learning_rate=4e-5, batch_size=64, target_update=2000, gamma=0.95, eps=1.0, eps_min=0.05, eps_period=50000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.target_model = DroneNet(n_actions=n_actions, state_vector_dim=model.vector_fc[0].in_features).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.memory = deque(maxlen=memory_size)
        
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = (eps - eps_min) / eps_period
        self.learn_step_counter = 0

    def get_action(self, img_state, vec_state, eps=None):
        current_eps = eps if eps is not None else self.eps
        if random.random() < current_eps:
            return None, random.randint(0, self.n_actions - 1)
        
        img_t = torch.FloatTensor(img_state).to(self.device)
        vec_t = torch.FloatTensor(vec_state).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(img_t, vec_t)
        
        self.model.train()
        
        q_array = q_values.cpu().numpy()[0]
        rounded_q = [round(float(val), 2) for val in q_array]
        return rounded_q, int(torch.argmax(q_values).item())

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"--- Đã lưu Model vào {filename} ---")

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"--- Đã tải Model thành công từ: {filename} ---")
