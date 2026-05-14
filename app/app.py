import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import json
import timm
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Dr. Diagnoser AI", page_icon="🔬", layout="wide")

# 2. Advanced Custom CSS (Fixed Layout)
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stApp { background-color: #f8fafc; }
    
    .report-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-top: 10px;
    }
    
    .main-title {
        color: #1e3a8a;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .badge {
        color: white;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 14px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# -------- 3. MODEL ARCHITECTURE --------
class ParallelBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1, self.b2, self.b3 = nn.Conv2d(3,64,3,padding=1), nn.Conv2d(3,64,5,padding=2), nn.Conv2d(3,64,7,padding=3)
    def forward(self,x):
        return torch.cat([torch.relu(self.b1(x)), torch.relu(self.b2(x)), torch.relu(self.b3(x))], dim=1)

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.parallel = ParallelBlock()
        self.conv1 = nn.Sequential(nn.Conv2d(192,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU())
    def forward(self,x):
        return self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(self.parallel(x))))))

class FinalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.cnn = DeepCNN()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512*4*4,512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512,128), nn.ReLU(), nn.Linear(128, num_classes))
    def forward(self,x):
        x = self.encoder(x).view(-1,3,16,16)
        return self.fc(self.cnn(x))

# -------- 4. ASSET LOADING --------
@st.cache_resource
def load_resources():
    with open("classes.json","r") as f: classes = json.load(f)
    model = FinalModel(num_classes=len(classes))
    model.load_state_dict(torch.load("final_model.pth", map_location="cpu"), strict=False)
    model.eval()
    return model, classes

model, classes = load_resources()

# -------- 5. UI HEADER & SIDEBAR --------
st.markdown("<h1 class='main-title'>🔬 Dr. Diagnoser <span style='color:#2563eb'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#64748b;'>Advanced Neural-Vision Hybrid for Multi-Cancer Diagnostics</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 50px;'>🩺</h1>", unsafe_allow_html=True)
    st.title("System Control")
    st.divider()
    st.success("● Engine: Online")
    st.info(f"Loaded Profiles: {len(classes)}")
    if st.button("Reset Analysis"): st.rerun()

# -------- 6. DASHBOARD LAYOUT --------
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📸 Patient Scan")
    uploaded = st.file_uploader("Upload MRI / CT / Histology", type=["jpg","png","jpeg"], label_visibility="collapsed")
    if uploaded:
        # Extra white box fix: Card sirf image ke saath load hoga
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        input_img = Image.open(uploaded).convert("RGB")
        st.image(input_img, caption="Input Stream", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("System waiting for diagnostic imagery...")

with col_result:
    if uploaded:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("⚡ Diagnostic Result")
        
        with st.status("Performing Neural Scan...", expanded=False) as status:
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])
            img_t = transform(input_img).unsqueeze(0)
            with torch.no_grad():
                out = model(img_t)
                prob = torch.softmax(out, dim=1)
                conf, pred = torch.max(prob, 1)
            status.update(label="Diagnostic Check Complete!", state="complete")

        # --- Label Formatting ---
        raw_label = classes[pred.item()].lower()
        display_name = raw_label.replace("_", " ").replace("std", "").title()
        is_negative = any(word in raw_label for word in ["normal", "healthy", "benign"])
        
        status_color, bg_color = ("#10b981", "#f0fff4") if is_negative else ("#ef4444", "#fff5f5")
        final_status = f"{display_name} [-ve]" if is_negative else f"{display_name} [+ve]"
        conf_score = conf.item() * 100

        # Result Presentation
        st.markdown(f"""
            <div style="text-align: center; padding: 25px; border-radius: 12px; background-color: {bg_color}; border: 2px solid {status_color};">
                <p style="margin: 0; color: #64748b; font-size: 0.9rem; font-weight: 600;">Diagnosis Insight</p>
                <h1 style="color: {status_color}; margin: 10px 0; font-size: 2.2rem;">{final_status}</h1>
                <span class="badge" style="background-color: {status_color};">Confidence: {conf_score:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Distribution Chart
        st.write("📊 **Neural Probability Distribution:**")
        prob_flat = prob.tolist()[0] # Fixing nested list for chart
        chart_data = pd.DataFrame({
            'Category': [c.replace("_", " ").title() for c in classes],
            'Confidence': [float(p) for p in prob_flat]
        }).sort_values(by='Confidence', ascending=False)
        st.bar_chart(chart_data, x='Category', y='Confidence', color="#3b82f6")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align:center; padding: 60px; color: #94a3b8; border: 2px dashed #cbd5e1; border-radius: 15px;">
                <h4>Waiting for Diagnostic Data...</h4>
                <p>System ready for automated multi-cancer scanning</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 30px;'>DR. DIAGNOSER AI v1.6 • RESEARCH TOOL</p>", unsafe_allow_html=True)
