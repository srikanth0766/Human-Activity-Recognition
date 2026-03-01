import torch
import torch.nn as nn
import torchvision.models as models

class SpatialBackbone(nn.Module):
    """
    ResNet-50 spatial feature extractor.
    Extracts 2048-d feature vectors from individual frames.
    """
    def __init__(self, pretrained=False):
        super(SpatialBackbone, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.avgpool = resnet.avgpool
        self.layer4 = resnet.layer4  # Keep accessible for Grad-CAM

    def forward(self, x):
        features = self.features(x)
        pooled = self.avgpool(features)
        flat = torch.flatten(pooled, 1)
        return flat

class TemporalModule(nn.Module):
    """Bidirectional LSTM for temporal sequence modeling."""
    def __init__(self, input_dim=2048, hidden_size=512, num_layers=2, dropout=0.3, bidirectional=True):
        super(TemporalModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.layer_norm(output)
        return output

class AttentionLayer(nn.Module):
    """Soft attention over temporal outputs."""
    def __init__(self, input_dim, attention_dim=256):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, lstm_output):
        scores = self.attention(lstm_output).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights

class ClassificationHead(nn.Module):
    """Classification head mapping to UCF50 classes."""
    def __init__(self, input_dim, num_classes=50, dropout=0.5):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),  # As specified in MEGA PROMPT: Dropout(0.4)
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)

class HARModel(nn.Module):
    """
    Person-Centric Spatio-Temporal HAR Model
    Input [B, T, C, H, W] → ResNet-50 → BiLSTM → Attention → Classification
    """
    def __init__(self, num_classes=50, pretrained_backbone=False):
        super(HARModel, self).__init__()
        self.num_classes = num_classes

        # 1. ResNet-50 backbone (pretrained=False when loading per mega prompt)
        self.backbone = SpatialBackbone(pretrained=pretrained_backbone)

        # 2. Extract 2048-d feature per frame, pass to BiLSTM
        self.temporal = TemporalModule(input_dim=2048, hidden_size=512, num_layers=2, dropout=0.3, bidirectional=True)
        
        temporal_output_dim = 512 * 2  # Bidirectional = True

        # 3. Attention layer over temporal dimension
        self.attention = AttentionLayer(input_dim=temporal_output_dim, attention_dim=256)
        
        # 4. Fully connected head to 50 classes
        self.classifier = ClassificationHead(input_dim=temporal_output_dim, num_classes=num_classes, dropout=0.4)

    def forward(self, x, return_attention=False):
        """
        x shape: (batch_size, seq_len, C, H, W)
        """
        batch_size, seq_len, C, H, W = x.shape

        frames = x.view(batch_size * seq_len, C, H, W)
        spatial_features = self.backbone(frames)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)

        temporal_output = self.temporal(spatial_features)
        context, attn_weights = self.attention(temporal_output)
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits
