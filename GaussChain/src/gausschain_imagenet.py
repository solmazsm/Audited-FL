# Author Solmaz Seyed Monir
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mpi4py import MPI
import time
import copy
from Dataset.ImageNet_LT_dataloader import ImageNetLTDataLoader
from lib_fl.models import modelC
from lib_fl.utils import average_weights

class GaussChainImageNet:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if args.gpu is not None else 'cpu'
        
        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Initialize dataset
        self.train_loader = ImageNetLTDataLoader(shuffle=True, training=True)
        self.test_loader = ImageNetLTDataLoader(shuffle=False, training=False)
        
        # Initialize model
        self.model = modelC(input_size=3, n_classes=1000).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.SGD(self.model.parameters(), 
                                 lr=args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize aggregation method
        self.aggr_method = args.aggr_method  # 'fedcls' or 'fedic'
        
    def train(self):
        for epoch in range(self.args.epochs):
            # Local training
            local_weights = self._local_training()
            
            # Global aggregation
            if self.aggr_method == 'fedcls':
                global_weights = self._fedcls_aggregation(local_weights)
            else:  # fedic
                global_weights = self._fedic_aggregation(local_weights)
            
            # Update global model
            self.model.load_state_dict(global_weights)
            
            # Evaluate
            if self.rank == 0:
                test_acc = self._evaluate()
                print(f'Epoch {epoch}: Test Accuracy = {test_acc:.4f}')
    
    def _local_training(self):
        self.model.train()
        epoch_loss = []
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Add Gaussian noise to gradients
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.args.gauss_var
                    param.grad += noise
            
            self.optimizer.step()
            epoch_loss.append(loss.item())
        
        return self.model.state_dict()
    
    def _fedcls_aggregation(self, local_weights):
        # FedCLS aggregation with class-balanced weights
        class_weights = self._compute_class_weights()
        weighted_weights = {}
        
        for key in local_weights.keys():
            weighted_weights[key] = local_weights[key] * class_weights
            
        return average_weights([weighted_weights])
    
    def _fedic_aggregation(self, local_weights):
        # FEDIC aggregation with importance sampling
        importance_weights = self._compute_importance_weights()
        weighted_weights = {}
        
        for key in local_weights.keys():
            weighted_weights[key] = local_weights[key] * importance_weights
            
        return average_weights([weighted_weights])
    
    def _compute_class_weights(self):
        # Compute class weights based on class distribution
        class_counts = torch.zeros(1000)
        for _, labels in self.train_loader:
            for label in labels:
                class_counts[label] += 1
        
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()
        return class_weights.to(self.device)
    
    def _compute_importance_weights(self):
        # Compute importance weights based on model performance
        self.model.eval()
        correct = torch.zeros(1000)
        total = torch.zeros(1000)
        
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(len(labels)):
                    label = labels[i]
                    total[label] += 1
                    if predicted[i] == label:
                        correct[label] += 1
        
        importance_weights = 1.0 - (correct / (total + 1e-6))
        importance_weights = importance_weights / importance_weights.sum()
        return importance_weights.to(self.device)
    
    def _evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total 
